#include "common.h"
#include <fftw3.h>

typedef std::complex<double> Complex;

Complex *srft_c, *w_c, *kw_c, *dft_c, *dct_c, *fwht_c, *dct_shift_c, *b_c;
int *bit_cnt, *bit_rev, *kbit_rev, k;
double *srft_re, *fwht_re, *b_re, *b_im;

void transpose(const double *a, double *temp, int N, int k) {
    int m = N / k;
#pragma omp for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            temp[i * k + j] = a[j * m + i];
}

void transpose(const Complex *a, Complex *temp, int N, int k) {
    int m = N / k;
#pragma omp for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            temp[i * k + j] = a[j * m + i];
}

void fft_parallel(Complex *a_c, int N, int k) {
    for (int i = 0; i < N / k; ++i) {
        fftw_plan plan = fftw_plan_dft_1d(
                k, reinterpret_cast<fftw_complex*>(a_c + i * k), reinterpret_cast<fftw_complex*>(b_c + i * k),
                FFTW_FORWARD,
                FFTW_ESTIMATE
        );
        fftw_execute(plan);
        fftw_destroy_plan(plan);
#pragma omp for
        for (int j = 0; j < k; ++j) a_c[j] = b_c[j];
    }
}

void dft_nlogd(Complex* a_c, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose(a_c, dft_c, N, k);
    fft_parallel(dft_c, N, k);
#pragma omp for
    for (int i = 0; i < d; ++i) {
        b_re[i] = 0;
        b_im[i] = 0;
    }
#pragma omp for collapse(2)
    for (int i = 0; i < d; ++i) {
        for (int p = 0; p < m; ++p) {
            int j = r[i];
            int y = j % k;
            // j * (p + q * m) / N = j * p / N + j * q / k
            // sum_q e^{2pi I j * (p + q * m) / N} = sum_q e^{2pi I (j * p / N) + (j * q / k)}
            // what we have: sum_q e^{2pi I y * q / k} for each y
            int x = ((int64_t)j * p) % N;
            Complex v = dft_c[p * k + y] * w_c[x];
            double re = v.real(), im = v.imag();
#pragma omp atomic
            b_re[i] += re;
#pragma omp atomic
            b_im[i] += im;
        }
    }
#pragma omp for
    for (int i = 0; i < d; ++i)
        a_c[i] = Complex(b_re[i], b_im[i]);
}

void dft(Complex *a_c, int N) {
    fft_parallel(a_c, N, N);
}

void dct(double *a, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        dct_c[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i];
    }
    fft_parallel(dct_c, N, N);
#pragma omp for
    for (int i = 0; i < N; ++i) {
        a[i] = (dct_c[i] * dct_shift_c[i]).real();
    }
}


void dct_nlogd(double *a, int N, int k, int d, const int *r) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        dct_c[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i];
    }
    dft_nlogd(dct_c, N, k, d, r);
#pragma omp for
    for (int i = 0; i < d; ++i) {
        int j = r[i];
        a[i] = (dct_shift_c[j] * dct_c[i]).real();
    }
}

/*
* @params N: array size
* @params d: r size
* @params n_ranks: num_ranks
* @params f: vector of N random signs (-1 or +1)
* @params perm: random permutation of [0, N)
* @params a: array to be srft'd
* @params r: d random elements from [0, N) (to be rd)
* @params sa_re: destination to store real part of result
* @params sa_im: destination to store imaginary part of result
* @params transform: the transformation to be performed
*/

void init(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform) {
    srft_c = new Complex[N];
    b_c = new Complex[N];
    if (transform == Transform::fourier || transform == Transform::cosine) {
        dft_c = new Complex[N];
    }
    if (transform == Transform::cosine) {
        dct_c = new Complex[N];
        dct_shift_c = new Complex[N + 1];
#pragma omp parallel for
        for (int i = 0; i <= N; ++i) {
            dct_shift_c[i] = Complex(2 * cos(M_PI * i / 2. / N), 2 * sin(M_PI * i / 2. / N));
        }
    }
    fftw_init_threads();
    fftw_plan_with_nthreads(n_ranks);
}

void init_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform) {
    k = 2;
    for (int i = 1; k < d * i && k < N; ++i) k *= 2;
    init(N, d, n_ranks, f, perm, r, transform);
}

void srft(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    if (transform == Transform::walsh || transform == Transform::cosine) {
#pragma omp for
        for (int i = 0; i < N; ++i) {
            srft_re[i] = a[perm[i]] * f[i];
        }
    } else {
#pragma omp for
        for (int i = 0; i < N; ++i) {
            srft_c[i] = a[perm[i]] * f[i];
        }
    }
    if (transform == Transform::walsh) {
        assert(false);
    } else if (transform == Transform::fourier) {
        dft(srft_c, N);
    } else {
        assert(transform == Transform::cosine);
        dct(srft_re, N);
    }
    double scale = sqrt(N / d);
    if (transform == Transform::walsh || transform == Transform::cosine) {
#pragma omp for
        for (int i = 0; i < d; ++i) {
            sa_re[i] = srft_c[r[i]].real() * scale;
            sa_im[i] = 0;
        }
    } else {
#pragma omp for
        for (int i = 0; i < d; ++i) {
            sa_re[i] = srft_c[r[i]].real() * scale;
            sa_im[i] = srft_c[r[i]].imag() * scale;
        }
    }
}

void srft_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    if (transform == Transform::walsh || transform == Transform::cosine) {
#pragma omp for
        for (int i = 0; i < N; ++i) {
            srft_re[i] = a[perm[i]] * f[i];
        }
    } else {
#pragma omp for
        for (int i = 0; i < N; ++i) {
            srft_c[i] = a[perm[i]] * f[i];
        }
    }
    if (transform == Transform::walsh) {
        assert(false);
    } else if (transform == Transform::fourier) {
        dft_nlogd(srft_c, N, k, d, r);
    } else {
        assert(transform == Transform::cosine);
        dct_nlogd(srft_re, N, k, d, r);
    }
    double scale = sqrt(N / d);
    if (transform == Transform::walsh || transform == Transform::cosine) {
#pragma omp for
        for (int i = 0; i < d; ++i) {
            sa_re[i] = srft_c[r[i]].real() * scale;
            sa_im[i] = 0;
        }
    } else {
#pragma omp for
        for (int i = 0; i < d; ++i) {
            sa_re[i] = srft_c[r[i]].real() * scale;
            sa_im[i] = srft_c[r[i]].imag() * scale;
        }
    }
}