#pragma once

#include <assert.h>
#include <iostream>

#include "rtree.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

template <typename T>
void box_d2h(BBOX<T>& h_box, const BBOX<T>& d_box) {
  int sz   = d_box.sz;
  h_box.sz = sz;

  h_box.xmin = new T[sz];
  assert(h_box.xmin != NULL);
  HANDLE_ERROR(cudaMemcpy(h_box.xmin, d_box.xmin, sz * sizeof(T), cudaMemcpyDeviceToHost));

  h_box.ymin = new T[sz];
  assert(h_box.ymin != NULL);
  HANDLE_ERROR(cudaMemcpy(h_box.ymin, d_box.ymin, sz * sizeof(T), cudaMemcpyDeviceToHost));

  h_box.xmax = new T[sz];
  assert(h_box.xmax != NULL);
  HANDLE_ERROR(cudaMemcpy(h_box.xmax, d_box.xmax, sz * sizeof(T), cudaMemcpyDeviceToHost));

  h_box.ymax = new T[sz];
  assert(h_box.ymax != NULL);
  HANDLE_ERROR(cudaMemcpy(h_box.ymax, d_box.ymax, sz * sizeof(T), cudaMemcpyDeviceToHost));

  h_box.id = new int[sz];
  assert(h_box.id != NULL);
  HANDLE_ERROR(cudaMemcpy(h_box.id, d_box.id, sz * sizeof(int), cudaMemcpyDeviceToHost));
}

template <typename T>
void box_h2d(BBOX<T>& d_box, const BBOX<T>& h_box) {
  int sz   = h_box.sz;
  d_box.sz = sz;

  HANDLE_ERROR(cudaMalloc((void**)&d_box.xmin, sz * sizeof(T)));
  assert(d_box.xmin != NULL);
  HANDLE_ERROR(cudaMemcpy(d_box.xmin, h_box.xmin, sz * sizeof(T), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void**)&d_box.ymin, sz * sizeof(T)));
  assert(d_box.ymin != NULL);
  HANDLE_ERROR(cudaMemcpy(d_box.ymin, h_box.ymin, sz * sizeof(T), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void**)&d_box.xmax, sz * sizeof(T)));
  assert(d_box.xmax != NULL);
  HANDLE_ERROR(cudaMemcpy(d_box.xmax, h_box.xmax, sz * sizeof(T), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void**)&d_box.ymax, sz * sizeof(T)));
  assert(d_box.ymax != NULL);
  HANDLE_ERROR(cudaMemcpy(d_box.ymax, h_box.ymax, sz * sizeof(T), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void**)&d_box.id, sz * sizeof(int)));
  assert(d_box.id != NULL);
  HANDLE_ERROR(cudaMemcpy(d_box.id, h_box.id, sz * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename T>
void box_d_free(BBOX<T>& box) {
  assert(box.xmin != NULL);
  cudaFree(box.xmin);
  box.xmin = NULL;

  assert(box.xmax != NULL);
  cudaFree(box.xmax);
  box.xmax = NULL;

  assert(box.ymin != NULL);
  cudaFree(box.ymin);
  box.ymin = NULL;

  assert(box.ymax != NULL);
  cudaFree(box.ymax);
  box.ymax = NULL;

  assert(box.id != NULL);
  cudaFree(box.id);
  box.id = NULL;
}

template <typename T>
void box_h_free(BBOX<T>& box) {
  assert(box.xmin != NULL);
  free(box.xmin);
  box.xmin = NULL;

  assert(box.xmax != NULL);
  free(box.xmax);
  box.xmax = NULL;

  assert(box.ymin != NULL);
  free(box.ymin);
  box.ymin = NULL;

  assert(box.ymax != NULL);
  free(box.ymax);
  box.ymax = NULL;

  assert(box.id != NULL);
  free(box.id);
  box.id = NULL;
}

template <typename T>
void rt_d_alloc(uint32_t sz, RTREE<T>& rt) {
  rt.sz = sz;

  HANDLE_ERROR(cudaMalloc((void**)&(rt.xmin), sz * sizeof(T)));
  assert(rt.xmin != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(rt.ymin), sz * sizeof(T)));
  assert(rt.ymin != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(rt.xmax), sz * sizeof(T)));
  assert(rt.xmax != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(rt.ymax), sz * sizeof(T)));
  assert(rt.ymax != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(rt.pos), sz * sizeof(int)));
  assert(rt.pos != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(rt.len), sz * sizeof(int)));
  assert(rt.len != NULL);
}

template <typename T>
void rt_d_free(RTREE<T>& rt) {
  assert(rt.xmin != NULL);
  cudaFree(rt.xmin);
  rt.xmin = NULL;

  assert(rt.ymin != NULL);
  cudaFree(rt.ymin);
  rt.ymin = NULL;

  assert(rt.xmax != NULL);
  cudaFree(rt.xmax);
  rt.xmax = NULL;

  assert(rt.ymax != NULL);
  cudaFree(rt.ymax);
  rt.ymax = NULL;

  assert(rt.len != NULL);
  cudaFree(rt.len);
  rt.len = NULL;

  assert(rt.pos != NULL);
  cudaFree(rt.pos);
  rt.pos = NULL;
}

void idpair_d_alloc(uint32_t sz, IDPAIR& ft) {
  ft.sz = sz;

  HANDLE_ERROR(cudaMalloc((void**)&(ft.fid), sz * sizeof(int)));
  assert(ft.fid != NULL);

  HANDLE_ERROR(cudaMalloc((void**)&(ft.tid), sz * sizeof(int)));
  assert(ft.tid != NULL);
}

void idpair_d2h(IDPAIR& h_ft, const IDPAIR& d_ft) {
  int sz  = d_ft.sz;
  h_ft.sz = sz;

  h_ft.fid = new int[sz];
  assert(h_ft.fid != NULL);
  HANDLE_ERROR(cudaMemcpy(h_ft.fid, d_ft.fid, sz * sizeof(int), cudaMemcpyDeviceToHost));

  h_ft.tid = new int[sz];
  assert(h_ft.tid != NULL);
  HANDLE_ERROR(cudaMemcpy(h_ft.tid, d_ft.tid, sz * sizeof(int), cudaMemcpyDeviceToHost));
}

void idpair_d_free(IDPAIR& ft) {
  assert(ft.fid != NULL);
  cudaFree(ft.fid);
  ft.fid = NULL;

  assert(ft.tid != NULL);
  cudaFree(ft.tid);
  ft.tid = NULL;
}

void idpair_h_free(IDPAIR& ft) {
  assert(ft.fid != NULL);
  free(ft.fid);
  ft.fid = NULL;

  assert(ft.tid != NULL);
  free(ft.tid);
  ft.tid = NULL;
}
