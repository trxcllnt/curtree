#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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
#include <time.h>
#include <iostream>
#include <vector>

#include "rtree.h"
#include "utility.cuh"
#include "utility.h"
#include "z_order.cuh"

using namespace std;

struct reduce_trans : public thrust::unary_function<int, int> {
  int d;
  reduce_trans(int _d) : d(_d) {}
  __device__ int operator()(int val) { return val / d; }
};

struct reduce_op {
  __device__ thrust::tuple<float, float, float, float, int, int> operator()(
    thrust::tuple<float, float, float, float, int, int> L,
    thrust::tuple<float, float, float, float, int, int> R) {
    float xmin = min(thrust::get<0>(L), thrust::get<0>(R));
    float ymin = min(thrust::get<1>(L), thrust::get<1>(R));
    float xmax = max(thrust::get<2>(L), thrust::get<2>(R));
    float ymax = max(thrust::get<3>(L), thrust::get<3>(R));
    int pos    = min(thrust::get<4>(L), thrust::get<4>(R));
    int count  = thrust::get<5>(L) + thrust::get<5>(R);
    return thrust::tuple<float, float, float, float, int, int>(xmin, ymin, xmax, ymax, pos, count);
  }
};

template <class T>
struct mbr2key : public thrust::unary_function<thrust::tuple<T, T, T, T>, uint> {
  __device__ uint operator()(thrust::tuple<T, T, T, T> t) {
    ushort x = (ushort)((thrust::get<0>(t) + thrust::get<2>(t)) / 2);
    ushort y = (ushort)((thrust::get<1>(t) + thrust::get<3>(t)) / 2);
    return z_order(x, y);
    // return (int)thrust::get<0>(t);
  }
};

template <class T>
int build_rtree(BBOX<T>& d_box, RTREE<T>& d_rt) {
  timeval t0, t1;
  thrust::device_ptr<T> d_xmin = thrust::device_pointer_cast(d_box.xmin);
  thrust::device_ptr<T> d_ymin = thrust::device_pointer_cast(d_box.ymin);
  thrust::device_ptr<T> d_xmax = thrust::device_pointer_cast(d_box.xmax);
  thrust::device_ptr<T> d_ymax = thrust::device_pointer_cast(d_box.ymax);
  thrust::device_ptr<int> d_id = thrust::device_pointer_cast(d_box.id);

  int size = d_box.sz;
  thrust::device_vector<int> d_sort_key(size);

  gettimeofday(&t0, NULL);

  // generate sort key
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_xmin, d_ymin, d_xmax, d_ymax)),
                    thrust::make_zip_iterator(thrust::make_tuple(
                      d_xmin + size, d_ymin + size, d_xmax + size, d_ymax + size)),
                    d_sort_key.begin(),
                    mbr2key<T>());

  thrust::sort_by_key(
    d_sort_key.begin(),
    d_sort_key.end(),
    thrust::make_zip_iterator(thrust::make_tuple(d_xmin, d_ymin, d_xmax, d_ymax, d_id)));
  gettimeofday(&t1, NULL);
  long sort_time = t1.tv_sec * 1000000 + t1.tv_usec - t0.tv_sec * 1000000 - t0.tv_usec;
  printf("sort time.......%10.2f\n", (float)sort_time / 1000);

  if (0) {
    std::cout << "in bulkload d_box.id" << std::endl;
    thrust::copy(d_id, d_id + size, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }

  // release sort key
  d_sort_key.resize(0);
  d_sort_key.shrink_to_fit();

  // pack R-tree_
  // first calculate levels
  vector<int> levels;
  int lev    = 0;
  int d      = d_rt.fanout;  // fanout
  int length = size;
  int R_sz   = 0;
  while (length != 1) {
    int num_nodes = ceil((float)length / d);
    R_sz += num_nodes;
    levels.push_back(num_nodes);
    length = num_nodes;
    lev++;
    cout << "level: " << lev << " len: " << length << endl;
  }

  rt_d_alloc(R_sz, d_rt);

  thrust::device_ptr<T> d_R_xmin  = thrust::device_pointer_cast(d_rt.xmin);
  thrust::device_ptr<T> d_R_ymin  = thrust::device_pointer_cast(d_rt.ymin);
  thrust::device_ptr<T> d_R_xmax  = thrust::device_pointer_cast(d_rt.xmax);
  thrust::device_ptr<T> d_R_ymax  = thrust::device_pointer_cast(d_rt.ymax);
  thrust::device_ptr<int> d_R_pos = thrust::device_pointer_cast(d_rt.pos);
  thrust::device_ptr<int> d_R_len = thrust::device_pointer_cast(d_rt.len);

  gettimeofday(&t0, NULL);
  int start_pos = R_sz;
  int next_sz   = size;
  thrust::constant_iterator<int> ONE(1);
  for (int i = 0; i < lev; i++) {
    start_pos -= levels[i];
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + next_sz;
    printf("lev=%d start_pos=%d\n", i, start_pos);
    if (i == 0)
      thrust::reduce_by_key(
        thrust::make_transform_iterator(first, reduce_trans(d)),
        thrust::make_transform_iterator(last, reduce_trans(d)),
        thrust::make_zip_iterator(thrust::make_tuple(
          d_xmin, d_ymin, d_xmax, d_ymax, thrust::make_counting_iterator<int>(R_sz), ONE)),
        thrust::make_discard_iterator(),
        thrust::make_zip_iterator(thrust::make_tuple(d_R_xmin + start_pos,
                                                     d_R_ymin + start_pos,
                                                     d_R_xmax + start_pos,
                                                     d_R_ymax + start_pos,
                                                     d_R_pos + start_pos,
                                                     d_R_len + start_pos)),
        thrust::equal_to<int>(),
        reduce_op());
    else
      thrust::reduce_by_key(thrust::make_transform_iterator(first, reduce_trans(d)),
                            thrust::make_transform_iterator(last, reduce_trans(d)),
                            thrust::make_zip_iterator(thrust::make_tuple(
                              d_R_xmin + start_pos + levels[i],
                              d_R_ymin + start_pos + levels[i],
                              d_R_xmax + start_pos + levels[i],
                              d_R_ymax + start_pos + levels[i],
                              thrust::make_counting_iterator<int>(start_pos + levels[i]),
                              ONE)),
                            thrust::make_discard_iterator(),
                            thrust::make_zip_iterator(thrust::make_tuple(d_R_xmin + start_pos,
                                                                         d_R_ymin + start_pos,
                                                                         d_R_xmax + start_pos,
                                                                         d_R_ymax + start_pos,
                                                                         d_R_pos + start_pos,
                                                                         d_R_len + start_pos)),
                            thrust::equal_to<int>(),
                            reduce_op());

    next_sz = levels[i];
  }
  assert(start_pos == 0);
  gettimeofday(&t1, NULL);
  long build_time = t1.tv_sec * 1000000 + t1.tv_usec - t0.tv_sec * 1000000 - t0.tv_usec;
  printf("build R tree time.......%10.2f\n", (float)build_time / 1000);
  cout << "total size: " << R_sz << " total level: " << lev << endl;

  if (0) {
    std::cout << "build: d_rt.pos=" << d_rt.pos << " d_rt.len=" << d_rt.len << std::endl;

    printf("level=%zu:", levels.size());
    thrust::copy(levels.begin(), levels.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    cout << "xmin:" << std::endl;
    thrust::copy(d_R_xmin, d_R_xmin + R_sz, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    cout << "ymin:" << std::endl;
    thrust::copy(d_R_ymin, d_R_ymin + R_sz, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    cout << "xmax:" << std::endl;
    thrust::copy(d_R_xmax, d_R_xmax + R_sz, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    cout << "ymax:" << std::endl;
    thrust::copy(d_R_ymax, d_R_ymax + R_sz, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    cout << "pos:" << std::endl;
    thrust::copy(d_R_pos, d_R_pos + R_sz, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    cout << "len:" << std::endl;
    thrust::copy(d_R_len, d_R_len + R_sz, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }
  if (0) {
    std::cout << "end bulkload d_box.id" << std::endl;
    thrust::device_ptr<int> t_id = thrust::device_pointer_cast(d_box.id);
    thrust::copy(t_id, t_id + size, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
  }

  return R_sz;
}
