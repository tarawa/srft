#ifndef _COMMON_H

#define _COMMON_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

enum class Transform {
    walsh,
    fourier,
    cosine
};

const double PI = acos(-1.0);

void init(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform);
void init_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform);
void srft(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *a_re, double *a_im, const int *r, Transform transform);
void srft_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *a_re, double *a_im, const int *r, Transform transform);

#endif