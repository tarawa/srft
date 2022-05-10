#include "common.h"
#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#define NUM_THREADS 256

typedef cufftDoubleComplex Complex;
typedef thrust::complex<double> complex_t;

Complex *a_c, *dct_c, *srft_c, *dct_shift_c, *w_c, *dft_c;
complex_t *d_c;
double *a_gpu, *sa_re_gpu, *sa_im_gpu;
int k;
int *f_gpu, *perm_gpu, *r_gpu;
cufftHandle planN, planK;

__global__ void transpose(const double *a, double *temp, int N, int k, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) temp[i] = a[(i % k) * m + (i / k)];
}

__global__ void transpose(const Complex *a, Complex *temp, int N, int k, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) temp[i] = a[(i % k) * m + (i / k)];
}

__global__ void compute_w(Complex* w_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= N) {
        w_c[i].x = cos(2 * M_PI * i / N);
        w_c[i].y = sin(2 * M_PI * i / N);
    }
}

__global__ void compute_dct_shift(Complex *dct_shift_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dct_shift_c[i].x = 2 * cos(M_PI * i / 2. / N);
    dct_shift_c[i].y = 2 * sin(M_PI * i / 2. / N);
}

__global__ void dft_nlogd_compute(complex_t *res_c, Complex *dft_c, Complex *w_c, int d, int m, const int *r, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d * m) return;
    int it = i / m, p = i % m;
    int j = r[it];
    int x = ((int64_t)j * p) % N, y = j % k;
    Complex v = dft_c[p * k + y];
    res_c[i] = complex_t(v.x * w_c[x].x - v.y * w_c[x].y, v.x * w_c[x].y + v.y * w_c[x].x);
}

__global__ void dft_nlogd_store(Complex *res_c, complex_t *d_c, int d, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    complex_t temp = d_c[i + m] - d_c[i];
    res_c[i].x = temp.real();
    res_c[i].y = temp.imag();
}

void dft_nlogd(Complex* a_c, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dft_c, N, k, m);
    if (cufftExecZ2Z(planK, dft_c, dft_c, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        assert(false);
    }
    dft_nlogd_compute<<<(d * m + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_c, dft_c, w_c, d, m, r_gpu, N, k);
    thrust::device_ptr<complex_t> d_c_ptr(d_c);
    thrust::inclusive_scan(d_c_ptr, d_c_ptr + d * m, d_c_ptr);
    dft_nlogd_store<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, d_c, d, m);
}

void dft(Complex *srft_c, int N) {
    if (cufftExecZ2Z(planN, srft_c, srft_c, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        assert(false);
    }
}

__global__ void dct_store(Complex *dct_c, Complex *a_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dct_c[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a_c[i];
}

__global__ void dct_load(Complex *a_c, Complex *dct_c, Complex *dct_shift_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    a_c[i].x = dct_c[i].x * dct_shift_c[i].x - dct_c[i].y * dct_shift_c[i].y;
    a_c[i].y = dct_c[i].x * dct_shift_c[i].y + dct_c[i].y * dct_shift_c[i].x;
}

void dct(Complex *a_c, int N) {
    dct_store<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dct_c, a_c, N);
    dft(dct_c, N);
    dct_load<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dct_c, dct_shift_c, N);
}

__global__ void dct_nlogd_load(Complex *a_c, Complex *dct_c, Complex *dct_shift_c, int d, const int *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    int j = r[i];
    a_c[i].x = dct_c[i].x * dct_shift_c[j].x - dct_c[i].y * dct_shift_c[j].y;
    a_c[i].y = dct_c[i].x * dct_shift_c[j].y + dct_c[i].y * dct_shift_c[j].x;
}

void dct_nlogd(Complex *a, int N, int k, int d, const int *r) {
    dct_store<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dct_c, a_c, N);
    dft_nlogd(dct_c, N, k, d, r);
    dct_nlogd_load<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dct_c, dct_shift_c, d, r_gpu);
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
    cudaMalloc((void**) &f_gpu, N * sizeof(int));
    cudaMalloc((void**) &perm_gpu, N * sizeof(int));
    cudaMalloc((void**) &r_gpu, d * sizeof(int));
    cudaMemcpy(f_gpu, f, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(perm_gpu, perm, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(r_gpu, r, d * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &a_gpu, N * sizeof(double));
    cudaMalloc((void**) &sa_re_gpu, d * sizeof(double));
    cudaMalloc((void**) &sa_im_gpu, d * sizeof(double));
    if (transform == Transform::walsh) {
        assert(false);
    }
    if (transform == Transform::fourier || transform == Transform::cosine) {
        if (cufftPlan1d(&planN, N, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
            assert(false);
        }
        cudaMalloc((void**) &dft_c, N * sizeof(Complex));
    }
    if (transform == Transform::cosine) {
        cudaMalloc((void**) &dct_c, N * sizeof(Complex));
        cudaMalloc((void**) &dct_shift_c, N * sizeof(Complex));
        compute_dct_shift<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dct_shift_c, N);
    }
}

void init_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform) {
    k = 2;
    for (int i = 1; k < d * i && k < N; ++i) k *= 2;
    init(N, d, n_ranks, f, perm, r, transform);
    if (transform == Transform::fourier || transform == Transform::cosine) {
        if (cufftPlan1d(&planK, k, CUFFT_Z2Z, N / k) != CUFFT_SUCCESS) {
            assert(false);
        }
        cudaMalloc((void**) &d_c, ((int64_t)d * (N / k)) * sizeof(complex_t));
        cudaMalloc((void**) &w_c, (N + 1) * sizeof(Complex));
        compute_w<<<(N + NUM_THREADS) / NUM_THREADS, NUM_THREADS>>>(w_c, N);
        cudaMemcpy(w_c + N, w_c, sizeof(Complex), cudaMemcpyDeviceToDevice);
    }
    if (transform == Transform::walsh) {
        assert(false);
    }
}

__global__ void shuffle(Complex *srft_c, double *a_gpu, int *perm_gpu, int *f_gpu, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    srft_c[i].x = a_gpu[perm_gpu[i]] * f_gpu[i];
    srft_c[i].y = 0;
}

__global__ void srft_save(double *sa_re_gpu, double *sa_im_gpu, double scale, Complex *srft_c, int d, const int *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    sa_re_gpu[i] = srft_c[r[i]].x;
    sa_im_gpu[i] = srft_c[r[i]].y;
}

__global__ void srft_save(double *sa_re_gpu, double *sa_im_gpu, double scale, Complex *srft_c, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    sa_re_gpu[i] = srft_c[i].x;
    sa_im_gpu[i] = srft_c[i].y;
}

void srft(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    cudaMemcpy(a_gpu, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (transform == Transform::walsh) {
        assert(false);
    } else {
        shuffle<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_c, a_gpu, perm_gpu, f_gpu, N);
        if (transform == Transform::fourier) {
            dft(srft_c, N);
        } else {
            assert(transform == Transform::cosine);
            dct(srft_c, N);
        }
        double scale = sqrt((double)N / d);
        srft_save<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(sa_re_gpu, sa_im_gpu, scale, srft_c, d, r_gpu);
        cudaMemcpy(sa_re, sa_re_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sa_im, sa_im_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

void srft_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    cudaMemcpy(a_gpu, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (transform == Transform::walsh) {
        assert(false);
    } else {
        shuffle<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_c, a_gpu, perm_gpu, f_gpu, N);
        if (transform == Transform::fourier) {
            dft_nlogd(srft_c, N, k, d, r);
        } else {
            assert(transform == Transform::cosine);
            dct_nlogd(srft_c, N, k, d, r);
        }
        double scale = sqrt((double)N / d);
        srft_save<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(sa_re_gpu, sa_im_gpu, scale, srft_c, d);
        cudaMemcpy(sa_re, sa_re_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sa_im, sa_im_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
    }
}
