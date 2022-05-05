#include <complex>
#include <cmath>
#include <iostream>
#include <complex>
#include <algorithm>

// #define Complex std::complex<double>
#define PI M_PI

void do_stuff(void);

void fwht(double* a, const int n);

void dft(double* a, const int n);

void fft(double* w, const int n);

void idft(double* a, const int n);

void srft(int n, int d, int* r, int* f, int* perm, double* a, double* sa, double* dest, void (*transform) (double*, int));

void fft_subsampled(double* a, int n, int d);

void srft_nlogd(int n, int d, int* r, int* f, int* perm, double* a, double* space, double* sa, void (*transform) (double*, int, int, int*, double*));

void dft_subsampled(double* v, int n, int d, int* perm, double* sa);

void fwht_subsampled(double* v, int n, int d, int* perm, double* sa);

static inline void transpose(double* a, int width, int height);

double dft_single(double* a, int m, int d, double* w, int index);
