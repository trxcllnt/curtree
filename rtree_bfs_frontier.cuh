#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<assert.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/sort.h>
#include<thrust/reduce.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/constant_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/iterator/discard_iterator.h>
#include<thrust/tuple.h>
#include<thrust/merge.h>
#include<thrust/remove.h>
#include<thrust/transform.h>
#include<iostream>
#include<vector>
#include<time.h>
#include<sys/time.h>


//#include "z_order.h"
#include "rtree.h"
#include "utility.h"
#include "utility.cuh"

using namespace std;

struct invalid_result
{
    __host__ __device__
    bool operator()(thrust::tuple<int, int> t)
    {
        return thrust::get<0>(t) ==-1;
    }
};

struct is_invalid
{
    __device__ __host__
    bool operator()(char flag)
    {
        return flag > 0;
    }
};

__host__ __device__ __inline__ bool intersect(float x1, float y1, float x2, float y2,
                          float m1, float n1, float m2, float n2)
{
    if (x1 > m2 || x2 < m1 || y1 > n2 || y2 < n1)
        return false;
    return true;    
}

#define BATCH 32 //number of query windows, i.e., 1*q_size; the larger, the higher parallelism and faster
#define QUEUE 1024 //number of maximum number of nodes after BFS expansion in a block

//NOTE: QUEUE is limited by shared memory capacity and needs to consider occupancy
// QUEUE/BATCH (32 in this particular case), should be the average expansion ratio for a block
//If any blocks has more than QUEUE children after expansion,i.e., an overflow happens, 
//a DFS phase (see https://dl.acm.org/doi/10.1145/2534921.2534949) 
//or a pure device memory based soultion needs to be applied

//TODO: allow speficy BATH and QUEUE from command line 

__global__ void query_tree_bfs_frontier(
                    int size,
                    int num_blocks,
                    float *q_xmin,
                    float *q_ymin,
                    float *q_xmax,
                    float *q_ymax,
                    float *R_xmin,
                    float *R_ymin,
                    float *R_xmax,
                    float *R_ymax,
                    int *R_pos,
                    int *R_len,
                    int R_size,
                    int *q_out_nodes,
                    int *q_out_ids,
                    char *q_out_flag
                    )
{
    __shared__ int nodes[QUEUE];
    __shared__ bool done;
    __shared__ int q_ids[QUEUE];
    __shared__ bool overflow;
    __shared__ int scan[QUEUE];
    int idx_block = blockIdx.x + blockIdx.y*gridDim.x;
    if (idx_block < num_blocks)
    {
        if(threadIdx.x == 0)
        {
            done = false;
            overflow = false;
        }
        __syncthreads();
        //init shared memory
        for (int i = 0; i < QUEUE/blockDim.x; i++)
            if (threadIdx.x+i*blockDim.x < QUEUE)
            {
                nodes[threadIdx.x+i*blockDim.x] = -1;
                q_ids[threadIdx.x+i*blockDim.x] = -1;
            }
        __syncthreads();
        //filling out the frontier queue
        if (threadIdx.x < BATCH && idx_block*BATCH+threadIdx.x < size)
        {
            nodes[threadIdx.x] = 0;
            q_ids[threadIdx.x] = idx_block*BATCH+threadIdx.x;
        }

        __syncthreads();
        while (!done&&!overflow)
        {
            //iter++;
            //each thread work on a node
            __syncthreads();
            int r_node = nodes[threadIdx.x];
            int id = q_ids[threadIdx.x];

            int children = 0;
            if (r_node >= 0 && id >= 0 && R_pos[r_node] < R_size)
            {
                nodes[threadIdx.x] = -1;
                q_ids[threadIdx.x] = -1;

                float xmin, ymin, xmax, ymax;
                xmin = q_xmin[id];
                ymin = q_ymin[id];
                xmax = q_xmax[id];
                ymax = q_ymax[id];

                for (int i = R_pos[r_node]; i < R_pos[r_node] + R_len[r_node]; i++)
                {
                    if (intersect(xmin, ymin, xmax, ymax,
                                R_xmin[i], R_ymin[i], R_xmax[i], R_ymax[i]))
                    {
                        children++;
                    }
                }
            }
            __syncthreads();
            //scan for position
            scan[threadIdx.x] = children;
            for (int offset = 1; offset < blockDim.x; offset <<= 1)
            {
                __syncthreads();
                //load
                int val = scan[threadIdx.x];
                if (threadIdx.x >= offset)
                    val += scan[threadIdx.x-offset];
                __syncthreads();
                //store
                scan[threadIdx.x] = val;
            }
            __syncthreads();
            
            //check overflow
            if (threadIdx.x == blockDim.x - 1)
            {
                if (scan[QUEUE-1] >= QUEUE)
                    overflow = true;
                if (scan[QUEUE-1] == 0)
                    done = true;
            }
            __syncthreads();
            if (!done && !overflow)
            {
                
                //write out next frontier
                if (children > 0)
                {
                    int pos = threadIdx.x == 0 ? 0 : scan[threadIdx.x-1];
                    float xmin, ymin, xmax, ymax;
                    xmin = q_xmin[id];
                    ymin = q_ymin[id];
                    xmax = q_xmax[id];
                    ymax = q_ymax[id];
                    for (int i = R_pos[r_node]; i < R_pos[r_node] + R_len[r_node]; i++)
                    {
                        if (intersect(xmin, ymin, xmax, ymax,
                                    R_xmin[i], R_ymin[i], R_xmax[i], R_ymax[i]))
                        {
                            nodes[pos] = i;
                            q_ids[pos] = id;
                            pos++;
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
        //write out results
        for (int i = 0; i < QUEUE/blockDim.x; i++)
            if (threadIdx.x+i*blockDim.x < QUEUE)
            {
                q_out_nodes[idx_block*QUEUE+threadIdx.x+i*blockDim.x] =  nodes[threadIdx.x+i*blockDim.x];
                q_out_ids[idx_block*QUEUE+threadIdx.x+i*blockDim.x]   =  q_ids[threadIdx.x+i*blockDim.x];
            }

        if (threadIdx.x == 0)
            if (overflow)
                q_out_flag[idx_block] = 1;
            else
                q_out_flag[idx_block] = 0;
    }
}

struct offset_functor
{
    int offset;
    int *pos;
    offset_functor(int *_pos, int _offset) : pos(_pos), offset(_offset){}
    __host__ __device__
    int operator()(int val)
    {
        return pos[val] - offset;
    }
};

struct count_hit
{
    float *rxmin, *rymin, *rxmax, *rymax;
    float *qxmin, *qymin, *qxmax, *qymax;
    int fanout, size;
    count_hit(float *q_x1, float *q_y1, float *q_x2, float *q_y2,
        float *r_x1, float *r_y1, float *r_x2, float *r_y2,int _size,int _fanout)
        : qxmin(q_x1), qymin(q_y1),qxmax(q_x2), qymax(q_y2), 
        rxmin(r_x1), rymin(r_y1), rxmax(r_x2), rymax(r_y2),size(_size),fanout(_fanout)
        {}
        
    __host__ __device__
    int operator()(thrust::tuple<int, int> t)
    {
        int hit = 0;
        int qid = thrust::get<0>(t);
        int rid = thrust::get<1>(t);
       
        for (int i = rid; i < fanout + rid; i++)
        {
            if (i >= size)
                break;
            if (intersect(qxmin[qid], qymin[qid], qxmax[qid], qymax[qid],
                    rxmin[i], rymin[i], rxmax[i], rymax[i]))
                hit++;           
        }
        return hit;
    }
};

struct write_hit
{
    float *rxmin, *rymin, *rxmax, *rymax;
    float *qxmin, *qymin, *qxmax, *qymax;
    int fanout, size;
    int *eid, *out_eid,*out_qid;
    write_hit(float *q_x1, float *q_y1, float *q_x2, float *q_y2,
        float *r_x1, float *r_y1, float *r_x2, float *r_y2,int *_eid,
        int _size,int _fanout,int *_out_eid,int *_out_qid)
        : qxmin(q_x1), qymin(q_y1),qxmax(q_x2), qymax(q_y2), 
        rxmin(r_x1), rymin(r_y1), rxmax(r_x2), rymax(r_y2),eid(_eid),
        size(_size),fanout(_fanout), out_eid(_out_eid), out_qid(_out_qid)        
        {}

    __host__ __device__
    int operator()(thrust::tuple<int, int, int> t)
    {
        int hit = 0;
        int qid = thrust::get<0>(t);
        int rid = thrust::get<1>(t);
        int pos = thrust::get<2>(t);
        
        for (int i = rid; i < fanout + rid; i++)
        {
            if (i >= size)
                break;
                
            if (intersect(qxmin[qid], qymin[qid], qxmax[qid], qymax[qid],
                    rxmin[i], rymin[i], rxmax[i], rymax[i]))
            {
                //printf("i=%d p=%d eid=%d qid=%d\n",i,pos+hit,eid[i],qid);
                out_eid[pos+hit] = eid[i];
                out_qid[pos+hit] = qid;
                hit++;           
            }
        }

        return hit;
    }
};

//d_dq: (d,q)<==>(f,t)
template <class T>
void bfs_frontier(const RTREE<T>& d_rt,const BBOX<T>& d_rbox,const BBOX<T>& d_qbox,IDPAIR& d_dq)
{
    assert(d_rt.pos!=NULL && d_rt.len!=NULL);
    int q_size=d_qbox.sz;
    int r_size=d_rt.sz;

if(0)
{
     std::cout<<"d_rt.pos="<<d_rt.pos<<" d_rt.len="<<d_rt.len<<std::endl;
     thrust::device_ptr<int> d_R_Pos=thrust::device_pointer_cast(d_rt.pos);
     thrust::device_ptr<int> d_R_Len=thrust::device_pointer_cast(d_rt.len);

    cout<<"bfs_frontier: pos:"<<std::endl;
    thrust::copy(d_R_Pos,d_R_Pos+r_size,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;
    cout<<"bfs_frontier: len:"<<std::endl;
    thrust::copy(d_R_Len,d_R_Len+r_size,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;

    thrust::device_ptr<T> d_R_xmin=thrust::device_pointer_cast(d_rt.xmin);
    thrust::device_ptr<T> d_R_ymin=thrust::device_pointer_cast(d_rt.ymin);
    thrust::device_ptr<T> d_R_xmax=thrust::device_pointer_cast(d_rt.xmax);
    thrust::device_ptr<T> d_R_ymax=thrust::device_pointer_cast(d_rt.ymax);
   
    cout<<"xmin:"<<std::endl;
    thrust::copy(d_R_xmin,d_R_xmin+r_size,std::ostream_iterator<float>(std::cout, " "));std::cout<<std::endl;
    cout<<"ymin:"<<std::endl;
    thrust::copy(d_R_ymin,d_R_ymin+r_size,std::ostream_iterator<float>(std::cout, " "));std::cout<<std::endl;
    cout<<"xmax:"<<std::endl;
    thrust::copy(d_R_xmax,d_R_xmax+r_size,std::ostream_iterator<float>(std::cout, " "));std::cout<<std::endl;
    cout<<"ymax:"<<std::endl;
    thrust::copy(d_R_ymax,d_R_ymax+r_size,std::ostream_iterator<float>(std::cout, " "));std::cout<<std::endl;        
}
    
    int num_threads = QUEUE;
    int num_blocks = ceil((float)q_size/(float)BATCH);
    int block_x = num_blocks;
    cout<<"num_blocks: "<<num_blocks<<" batch: "<<BATCH<<endl;
    int block_y = 1;

    if (num_blocks >= 65536)
    {
        block_x = 4096;
        block_y = ceil(num_blocks/4096.0f);
    }
    cout<<"block:  "<<block_x<<" "<<block_y<<endl;
    cout<<"thread: "<<num_threads<<endl;
    
    int *d_q_nodes;
    int *d_q_ids;
    char *d_q_out_flag;
    
    cudaMalloc((void **)&d_q_nodes, sizeof(int)*num_blocks*QUEUE);
    cudaMalloc((void **)&d_q_ids, sizeof(int)*num_blocks*QUEUE);
    cudaMalloc((void **)&d_q_out_flag, sizeof(char)*num_blocks);
    assert(d_q_nodes!=NULL && d_q_ids!=NULL && d_q_out_flag!=NULL);

    cout<<"start BFS...\n";
    cudaDeviceSynchronize();
    
    timeval t0, t1;
    gettimeofday(&t0, NULL);
    
    dim3 grid(block_x, block_y);
    dim3 block(num_threads);
        
    query_tree_bfs_frontier<<<grid, block>>>(q_size, num_blocks,
                 d_qbox.xmin, d_qbox.ymin, d_qbox.xmax, d_qbox.ymax,
                 d_rt.xmin,d_rt.ymin,d_rt.xmax,d_rt.ymax,
                 d_rt.pos, d_rt.len, r_size,
                 d_q_nodes, d_q_ids, d_q_out_flag);
    
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    long kernel_time = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
    printf("bfs kernel time.......%10.2f\n",(float)kernel_time/1000);

    //get indices for overflow blocks
    thrust::device_vector<int> invalid_blocks(num_blocks);
    thrust::device_ptr<char> d_q_out_flag_ptr(d_q_out_flag);
    
    int invalid_blocks_sz = 
    thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+num_blocks, 
                    d_q_out_flag_ptr, invalid_blocks.begin(), is_invalid()) - invalid_blocks.begin();

    invalid_blocks.resize(invalid_blocks_sz);
    cout<<"invalid block size: "<<invalid_blocks_sz<<endl;
    if(invalid_blocks_sz>0)
    {
        std::cout<<"shared memory overflew; dfs is needed.............."<<std::endl;
        exit(-1);
    }

    //remove invalid results from BFS results; 
    //this could be faster than the 2-phase counting/writting (node,query) pairs
    gettimeofday(&t0, NULL);
    int result_sz = num_blocks*QUEUE;
    thrust::device_ptr<int> d_q_nodes_ptr(d_q_nodes);
    thrust::device_ptr<int> d_q_ids_ptr(d_q_ids);
    result_sz = 
    thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(d_q_nodes_ptr, d_q_ids_ptr)),
                      thrust::make_zip_iterator(thrust::make_tuple(
                                                    d_q_nodes_ptr+result_sz,
                                                    d_q_ids_ptr+result_sz)),
                      //thrust::make_zip_iterator(thrust::make_tuple(d_q_nodes_ptr, d_q_ids_ptr)),
                      invalid_result()) - 
                      thrust::make_zip_iterator(thrust::make_tuple(d_q_nodes_ptr, d_q_ids_ptr));
    gettimeofday(&t1, NULL);
    long reduce_time = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
    printf("remove invalid results time.......%10.2f\n",(float)reduce_time/1000);
    cout<<"reduction size: "<<result_sz<<endl;

    //std::cout<<"before:"<<std::endl;
    //thrust::copy(d_q_nodes_ptr,d_q_nodes_ptr+result_sz,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    

    thrust::transform(d_q_nodes_ptr,d_q_nodes_ptr+result_sz,d_q_nodes_ptr, offset_functor(d_rt.pos, d_rt.sz));

    //std::cout<<"after:"<<std::endl;
    //thrust::copy(d_q_nodes_ptr,d_q_nodes_ptr+result_sz,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
   
    //(rtree node, query window) pair: (n,q)<==>(f,t)
    IDPAIR d_rq;
    idpair_d_alloc(result_sz,d_rq);
    HANDLE_ERROR( cudaMemcpy( d_rq.fid,d_q_nodes, result_sz* sizeof(T), cudaMemcpyDeviceToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_rq.tid,d_q_ids, result_sz* sizeof(T), cudaMemcpyDeviceToDevice ) );
     
    cudaFree(d_q_ids);
    cudaFree(d_q_nodes);
    cudaFree(d_q_out_flag);
    cudaDeviceSynchronize();
    
    thrust::device_vector<int> d_count(result_sz);
    thrust::device_ptr<int> rid_ptr=thrust::device_pointer_cast(d_rq.fid);
    thrust::device_ptr<int> qid_ptr=thrust::device_pointer_cast(d_rq.tid);

if(0)
{
    std::cout<<"rid_ptr:"<<std::endl;
    thrust::copy(rid_ptr,rid_ptr+result_sz,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
    std::cout<<"qid_ptr:"<<std::endl;
    thrust::copy(qid_ptr,qid_ptr+result_sz,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
    
}
    gettimeofday(&t0, NULL);
    //std::cout<<"d_rt.sz="<<d_rt.sz<<"d_rbox.sz="<<d_rbox.sz<<"  result_sz="<<result_sz<<std::endl;
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(qid_ptr, rid_ptr)),
            thrust::make_zip_iterator(thrust::make_tuple(qid_ptr+result_sz, rid_ptr+result_sz)),
            d_count.begin(), count_hit(
                                d_qbox.xmin, d_qbox.ymin,d_qbox.xmax,d_qbox.ymax,
                                d_rbox.xmin, d_rbox.ymin, d_rbox.xmax, d_rbox.ymax, 
                                d_rbox.sz,d_rt.fanout));

if(0)
{
    std::cout<<"counts:"<<std::endl;
    thrust::copy(d_count.begin(),d_count.end(),std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
}

    cout<<"prefix scan..."<<endl;
    int total_hit = d_count.back();
    thrust::exclusive_scan(d_count.begin(), d_count.end(), d_count.begin());
    total_hit += d_count.back();
    cout<<"total hit: "<<total_hit<<endl;

if(0)
{
    std::cout<<"rbox.id"<<std::endl;
    thrust::device_ptr<int> rid_ptr=thrust::device_pointer_cast(d_rbox.id);
    thrust::copy(rid_ptr,rid_ptr+d_rbox.sz,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
}

    //(data, query window) pair: (d,q)<==>(f,t)
    idpair_d_alloc(total_hit,d_dq);    
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(qid_ptr,  rid_ptr, d_count.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(qid_ptr+result_sz, rid_ptr+result_sz, d_count.end())),
            thrust::make_discard_iterator(), write_hit(d_qbox.xmin, d_qbox.ymin,d_qbox.xmax,d_qbox.ymax,
                                d_rbox.xmin, d_rbox.ymin, d_rbox.xmax, d_rbox.ymax, d_rbox.id,d_rbox.sz,
                                d_rt.fanout, d_dq.fid, d_dq.tid));
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    long write_time = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
    printf("write results time.......%10.2f\n",(float)write_time/1000);

if(0)
{
     thrust::device_ptr<int> fid_ptr=thrust::device_pointer_cast(d_dq.fid);
     thrust::device_ptr<int> tid_ptr=thrust::device_pointer_cast(d_dq.tid);
     
     std::cout<<"fid_ptr:"<<std::endl;
     thrust::copy(fid_ptr,fid_ptr+total_hit,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
     std::cout<<"tid_ptr:"<<std::endl;
     thrust::copy(tid_ptr,tid_ptr+total_hit,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;    
}

}
