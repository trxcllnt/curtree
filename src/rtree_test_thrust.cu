/*
    nvcc -O3 rtree_test_thrust.cu utility.cpp -o rtree_test_thrust

   ./rtree_test_thrust.out input_44edges.txt 44  input_44edges.txt  44 3
   ./rtree_test_thrust input_39096edges.txt 39096 input_39096edges.txt 39096 10
   ./rtree_test_thrust edges.txt 1154548 edges.txt 1154548 10
*/
//#include "z_order.h"

#include "rmm/mr/device/device_memory_resource.hpp"
#include "rtree.h"
#include "rtree_bfs_thrust.cuh"
#include "rtree_bulkload_thrust.cuh"
#include "utility.cuh"
#include "utility.h"

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>

using namespace std;

namespace {
// memory resource factory helpers
auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

template <typename Upstream>
auto make_pool(std::shared_ptr<Upstream> mr) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(mr);
}
}  // namespace

int main(int argc, char *argv[]) {
  auto mr = make_cuda();
  // auto mr = make_pool(make_cuda());
  rmm::mr::set_current_device_resource(mr.get());

  if (argc != 6) {
    printf("EXEC input_mbr input_num, query_mbr, query_num, rtree_fanout\n");
    return -1;
  }

  printf("input MBR file name: %s\n", argv[1]);
  printf("input number of MBRs: %s\n", argv[2]);
  printf("query MBR file name: %s\n", argv[3]);
  printf("query number of MBRs: %s\n", argv[4]);
  printf("rtree fanout: %s\n", argv[5]);

  using T = float;

  BBOX<T> h_d_box, d_r_box;
  BBOX<T> h_q_box, d_q_box;
  RTREE<T> d_rt;

  h_d_box.sz  = atoi(argv[2]);
  h_q_box.sz  = atoi(argv[4]);
  d_rt.fanout = atoi(argv[5]);

  // currently readMBR handles only float type
  readMBR(
    argv[1], h_d_box.sz, 0, h_d_box.xmin, h_d_box.ymin, h_d_box.xmax, h_d_box.ymax, h_d_box.id);
  box_h2d(d_r_box, h_d_box);

  build_rtree(d_r_box, d_rt);

  readMBR(
    argv[3], h_q_box.sz, 10, h_q_box.xmin, h_q_box.ymin, h_q_box.xmax, h_q_box.ymax, h_q_box.id);

  if (0) {
    std::cout << "h_q_box.id" << std::endl;
    thrust::copy(h_q_box.id, h_q_box.id + h_q_box.sz, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
  box_h2d(d_q_box, h_q_box);

  IDPAIR d_dq, h_dq;

  // only generating (node, query) pairs for now
  bfs_thrust(d_rt, d_r_box, d_q_box, d_dq);

  idpair_d2h(h_dq, d_dq);

  // first sort by data box id
  thrust::stable_sort_by_key(h_dq.fid, h_dq.fid + h_dq.sz, h_dq.tid);
  // then sort by query box id
  thrust::stable_sort_by_key(h_dq.tid, h_dq.tid + h_dq.sz, h_dq.fid);

  thrust::host_vector<int> h_g_did(h_dq.fid, h_dq.fid + h_dq.sz);
  thrust::host_vector<int> h_g_qid(h_dq.tid, h_dq.tid + h_dq.sz);

  // print_result_list(h_g_did,h_g_qid);

  // verification
  thrust::host_vector<int> h_c_did, h_c_qid;
  int num_h = nested_loop_query<float>(h_q_box, h_d_box, h_c_did, h_c_qid);
  std::cout << "num_h=" << num_h << std::endl;

  bool same_d = same_vector(-1, h_g_did, h_c_did);
  bool same_q = same_vector(-1, h_g_qid, h_c_qid);

  if (same_d && same_q) {
    printf("verified successfully\n");
  } else {
    printf("same box ids=%d same query ids=%d\n", same_d, same_q);
    printf("verification failed\n");
  }
  box_d_free(d_r_box);
  box_h_free(h_d_box);

  box_d_free(d_q_box);
  box_h_free(h_q_box);

  rt_d_free(d_rt);

  idpair_d_free(d_dq);
  idpair_h_free(h_dq);

  return 0;
}
