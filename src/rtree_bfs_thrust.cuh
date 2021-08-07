#pragma once

#include "rmm/detail/error.hpp"
#include "rtree.h"
#include "thrust/count.h"
#include "thrust/distance.h"
#include "thrust/iterator/permutation_iterator.h"
#include "utility.cuh"
#include "utility.h"

#include <ratio>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
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
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace thrust::placeholders;

__host__ __device__ __inline__ bool intersect(
  float x1, float y1, float x2, float y2, float m1, float n1, float m2, float n2) {
  return !(x1 > m2 || x2 < m1 || y1 > n2 || y2 < n1);
}

template <typename T>
inline rmm::device_uvector<T> make_and_initialize_device_uvector(size_t size,
                                                                 T const default_value,
                                                                 rmm::cuda_stream_view stream) {
  auto vec = rmm::device_uvector<T>(size, stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream), vec.begin(), vec.end(), default_value);
  return vec;
}

// d_dq: (box,query)<==>(f,t)
template <class T>
void bfs_thrust(const RTREE<T>& d_rt, const BBOX<T>& d_rbox, const BBOX<T>& d_qbox, IDPAIR& d_dq) {
  int32_t q_size    = d_qbox.sz;
  int32_t work_size = q_size * 1;
  auto stream       = rmm::cuda_stream_default;

  thrust::device_ptr<int32_t> d_rt_pos = thrust::device_pointer_cast(d_rt.pos);
  thrust::device_ptr<int32_t> d_rt_len = thrust::device_pointer_cast(d_rt.len);

  auto qbox_ids = rmm::device_uvector<int32_t>(work_size, stream);
  // root id always begin with position 0
  auto node_ids = make_and_initialize_device_uvector<int32_t>(work_size, 0, stream);
  // query box ids start out sequential
  thrust::sequence(rmm::exec_policy(stream), qbox_ids.begin(), qbox_ids.end());

  auto tree_boxes  = thrust::make_zip_iterator(d_rt.xmin, d_rt.ymin, d_rt.xmax, d_rt.ymax);
  auto query_boxes = thrust::make_zip_iterator(d_qbox.xmin, d_qbox.ymin, d_qbox.xmax, d_qbox.ymax);

  auto is_non_leaf_intersection = [rt_sz = d_rt.sz] __device__(auto const& tup) {
    auto& nid  = thrust::get<0>(tup);
    auto& qbox = thrust::get<1>(tup);
    auto& node = thrust::get<2>(tup);
    return (nid >= 0 && nid < rt_sz) && intersect(thrust::get<0>(qbox),
                                                  thrust::get<1>(qbox),
                                                  thrust::get<2>(qbox),
                                                  thrust::get<3>(qbox),
                                                  thrust::get<0>(node),
                                                  thrust::get<1>(node),
                                                  thrust::get<2>(node),
                                                  thrust::get<3>(node));
  };

  auto start_time = std::chrono::high_resolution_clock::now();

  auto last_work_size = work_size;

  for (int32_t lev = 0, valid_size = work_size; work_size > 0 && valid_size > 0; lev++) {
    // std::cout << "lev = " << lev << ", "
    //           << "work_size = " << work_size << ", "
    //           << "valid_size = " << valid_size << std::endl;

    // expand for next loop
    auto d_map = rmm::device_uvector<int32_t>(valid_size + 1, stream);
    thrust::inclusive_scan(
      rmm::exec_policy(stream),
      thrust::make_permutation_iterator(d_rt_len, node_ids.begin()),
      thrust::make_permutation_iterator(d_rt_len, node_ids.begin()) + valid_size,
      d_map.begin() + 1);

    work_size = d_map.back_element(stream);  // synchronizes the stream

    // std::cout << "lev=" << lev << " next_size=" << next_size << std::endl;

    if (work_size > 0) {
      last_work_size = work_size;

      // initialize d_map[0] to 0
      d_map.set_element_to_zero_async(0, stream);

      // Wrapped in an IEFE so intermediate memory is freed
      std::tie(qbox_ids, node_ids) = [&] {
        // calculate next qbox ids and node ids
        auto next_qbox_ids = make_and_initialize_device_uvector<int32_t>(work_size, -1, stream);
        auto next_node_ids = make_and_initialize_device_uvector<int32_t>(work_size, -1, stream);
        auto offsets       = make_and_initialize_device_uvector<int32_t>(work_size, -1, stream);

        auto offsets_and_qbox_id_and_node_pos =
          thrust::make_zip_iterator(d_map.begin(),
                                    qbox_ids.begin(),
                                    thrust::make_permutation_iterator(d_rt_pos, node_ids.begin()));

        auto offsets_and_next_qbox_and_node_ids =
          thrust::make_zip_iterator(offsets.begin(), next_qbox_ids.begin(), next_node_ids.begin());

        // fused scatter to:
        // * scatter map into offsets for later
        // * scatter id vectors into next id vectors
        thrust::scatter(rmm::exec_policy(stream),
                        offsets_and_qbox_id_and_node_pos,
                        offsets_and_qbox_id_and_node_pos + valid_size,
                        d_map.begin(),
                        offsets_and_next_qbox_and_node_ids);

        auto next_qbox_and_offset =
          thrust::make_zip_iterator(next_qbox_ids.begin(), offsets.begin());

        // fused inclusive_scan to:
        // * calculate next qbox ids
        // * calculate offsets so we can rebase node ids
        thrust::inclusive_scan(rmm::exec_policy(stream),
                               next_qbox_and_offset,
                               next_qbox_and_offset + work_size,
                               next_qbox_and_offset,
                               [] __device__(auto const& a, auto const& b) {
                                 return thrust::make_tuple(
                                   thrust::max(thrust::get<0>(a), thrust::get<0>(b)),
                                   thrust::max(thrust::get<1>(a), thrust::get<1>(b)));
                               });

        // calculate next node ids
        thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                      next_qbox_ids.begin(),
                                      next_qbox_ids.begin() + work_size,
                                      next_node_ids.begin(),
                                      next_node_ids.begin(),
                                      thrust::equal_to<int32_t>(),
                                      thrust::maximum<int32_t>());

        // rebase the next node ids
        auto idx_offset_and_node_id = thrust::make_zip_iterator(
          thrust::make_counting_iterator(0), offsets.begin(), next_node_ids.begin());

        thrust::transform(rmm::exec_policy(stream),
                          idx_offset_and_node_id,
                          idx_offset_and_node_id + work_size,
                          next_node_ids.begin(),
                          [] __device__(auto const& tup) {
                            auto index   = thrust::get<0>(tup);
                            auto offset  = thrust::get<1>(tup);
                            auto node_id = thrust::get<2>(tup);
                            return (index - offset) + node_id;
                          });

        return std::make_pair(std::move(next_qbox_ids), std::move(next_node_ids));
      }();

      auto qbox_and_node_ids = thrust::make_zip_iterator(qbox_ids.begin(), node_ids.begin());

      auto non_leaf_intersections = thrust::make_transform_iterator(
        thrust::make_zip_iterator(node_ids.begin(),
                                  thrust::make_permutation_iterator(query_boxes, qbox_ids.begin()),
                                  thrust::make_permutation_iterator(tree_boxes, node_ids.begin())),
        is_non_leaf_intersection);

      // Remove the non-intersection or leaf query and node ids from consideration
      valid_size = thrust::distance(qbox_and_node_ids,
                                    thrust::copy_if(rmm::exec_policy(stream),
                                                    qbox_and_node_ids,
                                                    qbox_and_node_ids + work_size,
                                                    non_leaf_intersections,
                                                    qbox_and_node_ids,
                                                    thrust::identity<bool>()));
    }
  }

  std::cout << "final work_size = " << last_work_size << std::endl;

  // remove (qid, boxid) pairs that do not intersect
  auto qbox_and_node_ids = thrust::make_zip_iterator(qbox_ids.begin(), node_ids.begin());
  auto result_size =
    thrust::distance(qbox_and_node_ids,
                     thrust::copy_if(rmm::exec_policy(stream),
                                     qbox_and_node_ids,
                                     qbox_and_node_ids + last_work_size,
                                     qbox_and_node_ids,
                                     [offset    = d_rt.sz,
                                      qbox_xmin = d_qbox.xmin,
                                      qbox_ymin = d_qbox.ymin,
                                      qbox_xmax = d_qbox.xmax,
                                      qbox_ymax = d_qbox.ymax,
                                      rbox_xmin = d_rbox.xmin,
                                      rbox_ymin = d_rbox.ymin,
                                      rbox_xmax = d_rbox.xmax,
                                      rbox_ymax = d_rbox.ymax] __device__(auto const& pair) {
                                       auto qbox_id = thrust::get<0>(pair);
                                       auto dbox_id = thrust::get<1>(pair) - offset;
                                       return intersect(qbox_xmin[qbox_id],
                                                        qbox_ymin[qbox_id],
                                                        qbox_xmax[qbox_id],
                                                        qbox_ymax[qbox_id],
                                                        rbox_xmin[dbox_id],
                                                        rbox_ymin[dbox_id],
                                                        rbox_xmax[dbox_id],
                                                        rbox_ymax[dbox_id]);
                                     }));

  std::cout << "result_size = " << result_size << std::endl;

  idpair_d_alloc(result_size, d_dq);
  thrust::device_ptr<int32_t> d_nid_ptr = thrust::device_pointer_cast(d_dq.fid);
  thrust::device_ptr<int32_t> d_qid_ptr = thrust::device_pointer_cast(d_dq.tid);

  // substract offset so that box id begins with 0 (originally relative to root node position)
  thrust::transform(rmm::exec_policy(stream),
                    qbox_and_node_ids,
                    qbox_and_node_ids + result_size,
                    thrust::make_zip_iterator(d_qid_ptr, d_nid_ptr),
                    [offset = d_rt.sz, ids = d_rbox.id] __device__(auto const& pair) {
                      auto& qbox_id = thrust::get<0>(pair);
                      auto& node_id = thrust::get<1>(pair);
                      return thrust::make_tuple(qbox_id, ids[node_id - offset]);
                    });

  auto run_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now() - start_time);

  std::cout << "thrust bfs end-to-end time: " << (run_time.count() / 1000000.) << "ms" << std::endl;
}
