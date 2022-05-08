#ifndef _COMMON_H

#define _COMMON_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

enum class Transform {
    walsh,
    fourier,
    cosine
};

const double PI = acos(-1.0);

void srft(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform);
void srft_nlogd(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform);

#endif