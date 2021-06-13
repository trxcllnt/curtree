#pragma once

#include<thrust/host_vector.h>

void readMBR(char *file, const int size,const float r,
    float* &xmin, float* &ymin,float* &xmax, float* &ymax, int* &id);

template<class T>
inline bool intersect_cpu(T x1, T y1, T x2, T y2,
                          T m1, T n1, T m2, T n2)
{
    if (x1 > m2 || x2 < m1 || y1 > n2 || y2 < n1)
        return false;
    return true;
}


template<class T>
int nested_loop_query(const BBOX<T>& q_box, const BBOX<T>& d_box,
    thrust::host_vector<int>& v_did,thrust::host_vector<int>& v_qid)
{

    for(int i=0;i<q_box.sz;i++)//query
    {
        for(int j=0;j<d_box.sz;j++)//data
        {
            if (intersect_cpu(q_box.xmin[i],q_box.ymin[i],q_box.xmax[i],q_box.ymax[i],
                d_box.xmin[j],d_box.ymin[j],d_box.xmax[j],d_box.ymax[j]))
            {
               v_did.push_back(d_box.id[j]);
               v_qid.push_back(q_box.id[i]);
            }
        }
     }
    return(v_did.size());
}

bool same_vector(int num_print,const thrust::host_vector<int>& s,const thrust::host_vector<int>& d)
{
	if(num_print>0)
	{
		printf("source ids:\n");
		thrust::copy(s.begin(),s.begin()+num_print,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;
	    printf("d_pid:\n");
		thrust::copy(d.begin(),d.begin()+num_print,std::ostream_iterator<int>(std::cout, " "));std::cout<<std::endl;
	 }
    bool same=thrust::equal(thrust::host,s.begin(),s.end(), d.begin());

    if(!same)
    {
     printf("num_s=%zu num_q=%zu\n",s.size(),d.size());
     assert(s.size()==d.size());
     for(int i=0;i<s.size();i++)
         if(s[i]!=d[i])
             printf("%d %u %u\n",i,s[i],d[i]);
     printf("---------------------------------------------\n");
    }

   return(same);
}

//assume pid and qid are sorted first based on pid and then qid so that one unique qid is associated wiht multiple pid
int print_result_list(const thrust::host_vector<int>& eid,const thrust::host_vector<int>& qid)
{
   int sz=eid.size();
   assert(sz==qid.size());
   thrust::host_vector<int> key(sz);
   thrust::host_vector<int> num(sz);
   int ne=thrust::reduce_by_key(thrust::host,qid.begin(),qid.end(),thrust::constant_iterator<int>(1),key.begin(),num.begin()).first-key.begin();

 if(1)
 {
    printf("num_e=%u\n",ne);
    int p=0;
    for(int i=0;i<ne;i++)
    {
       printf("(%u %u)==> ",key[i],num[i]);
       for(int j=0;j<num[i];j++)
         printf("%u,",eid[p++]);
       printf("\n");
    }
    printf("p=%u\n",p);
    return p;
 }

}


