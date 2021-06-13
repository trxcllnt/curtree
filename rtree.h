#pragma once

template<class T>
struct BBOX
{
	int sz;
	T* xmin;
	T* ymin;
	T* xmax;
	T* ymax;
	int *id;
};

template<class T>
struct RTREE
{
	int sz;
	int fanout;
	T* xmin;
	T* ymin;
	T* xmax;
	T* ymax;
	int *pos;
	int *len;
};

struct IDPAIR
{
	int sz;
	int *fid;
	int *tid;
};

