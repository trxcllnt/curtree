//nvcc rtree_test_frontier.cu utility.cpp

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
#include "rtree_bulkload_thrust.cuh"
#include "rtree_bfs_frontier.cuh"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("EXEC input_mbr fanout query_mbr\n");
        return -1;
    }
    
    using T=float;
    
    BBOX<T> h_d_box,d_r_box;
    h_d_box.sz=44;
    
    //currently readMBR handles only float type
    readMBR(argv[1],h_d_box.sz,0,h_d_box.xmin, h_d_box.ymin, h_d_box.xmax, h_d_box.ymax, h_d_box.id); 
    box_h2d(d_r_box,h_d_box);
    
    RTREE<T> d_rt;
    int fanout=atoi(argv[2]);
    d_rt.fanout=fanout;
    build_rtree(d_r_box,d_rt);
    std::cout<<"main:   d_rt.pos="<<d_rt.pos<<" d_rt.len="<<d_rt.len<<std::endl;

    BBOX<T> h_q_box,d_q_box;
    h_q_box.sz=44;    
    readMBR(argv[3],h_q_box.sz,10,h_q_box.xmin, h_q_box.ymin, h_q_box.xmax, h_q_box.ymax, h_q_box.id);
    box_h2d(d_q_box,h_q_box);

    IDPAIR d_dq,h_dq;
    
    bfs_frontier(d_rt,d_r_box,d_q_box,d_dq);
    
    idpair_d2h(h_dq,d_dq);
    
    //first sort by data box id
    thrust::stable_sort_by_key(h_dq.fid,h_dq.fid+h_dq.sz,h_dq.tid);
    //then sort by query box id
    thrust::stable_sort_by_key(h_dq.tid,h_dq.tid+h_dq.sz,h_dq.fid);
    
    thrust::host_vector<int> h_g_did(h_dq.fid,h_dq.fid+h_dq.sz);
    thrust::host_vector<int> h_g_qid(h_dq.tid,h_dq.tid+h_dq.sz);
        
    print_result_list(h_g_did,h_g_qid);
    
    //verification
    thrust::host_vector<int> h_c_did,h_c_qid;
    int num_h=nested_loop_query<float>(h_q_box, h_d_box,h_c_did,h_c_qid);
    std::cout<<"num_h="<<num_h<<std::endl;
 
    bool same_d=same_vector(-1,h_g_did,h_c_did);
    bool same_q=same_vector(-1,h_g_qid,h_c_qid);
    
    if(same_d && same_q)
        printf("verified successfully\n");
    else
        printf("verification failed\n");

    box_d_free(d_r_box);
    box_h_free(h_d_box);
    
    box_d_free(d_q_box);
    box_h_free(h_q_box);
    
    rt_d_free(d_rt);
    
    idpair_d_free(d_dq);
    idpair_h_free(h_dq);

    return 0;
    
}    