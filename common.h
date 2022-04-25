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

