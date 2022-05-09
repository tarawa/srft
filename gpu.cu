#include "common.h"
#include <cuda.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#define NUM_THREADS 256

typedef thrust::complex<double> Complex;

Complex *srft_c, *w_c, *kw_c, *dft_c, *dct_c, *fwht_c, *a_c, *dct_shift_c, *b_c, *d_c;
double *fwht_r, *a_gpu, *sa_re_gpu, *sa_im_gpu, *d_r, *srft_r;
int *bit_cnt, *bit_rev, *kbit_rev, k;
int *f_gpu, *perm_gpu, *r_gpu;

bool flag;

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
    if (i <= N) w_c[i] = Complex(cos(2 * M_PI * i / N), sin(2 * M_PI * i / N));
}

__global__ void compute_bit_rev(int* bit_rev, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = 0;
    for (int x = (N >> 1), y = i; x; x >>= 1, y >>= 1) j = ((j << 1) | (y & 1));
    bit_rev[i] = j;
}

__global__ void compute_bitcount(int *bit_cnt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int temp = (i & 0x55555555) + ((i >> 1) & 0x55555555);
    temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
    temp = (temp & 0x0F0F0F0F) + ((temp >> 4) & 0x0F0F0F0F);
    temp = (temp & 0x00FF00FF) + ((temp >> 8) & 0x00FF00FF);
    temp = (temp & 0x0000FFFF) + ((temp >> 16) & 0x0000FFFF);
    bit_cnt[i] = temp;
}

__global__ void compute_dct_shift(Complex *dct_shift_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dct_shift_c[i] = Complex(2 * cos(PI * i / 2. / N), 2 * sin(PI * i / 2. / N));
}

__global__ void fwht_butterfly(double *a_gpu, int N, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N / 2) return;
    int p = (i / h) * h * 2, q = i % h;
    double x = a_gpu[p + q], y = a_gpu[p + q + h];
    a_gpu[p + q] = x + y;
    a_gpu[p + q + h] = x - y;
}

__device__ void fwht(double* a, int N) {
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += h * 2) {
            for (int j = 0; j < h; ++j) {
                double x = a[i + j];
                double y = a[i + j + h];
                a[i + j] = x + y;
                a[i + j + h] = x - y;
            }
        }
    }
}

void fwht_parallel(double* a, int N) {
    for (int h = 1; h < N; h *= 2) {
        fwht_butterfly<<<(N / 2 + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a, N, h);
    }
}

__global__ void fwht_block(double *a, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * k >= N) return;
    fwht(a + i * k, k);
}

__global__ void fwht_nlogd_compute(double *d_r, double *fwht_r, int d, int m, const int *r, int k, const int *bit_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d * m) return;
    int it = i / m, p = i % m;
    int j = r[it], y = j / m;
    double v = fwht_r[p * k + y];
    double x = (bit_cnt[j & p] ? -v : v);
    d_r[i] = x;
}

__global__ void fwht_nlogd_store(double *res_r, double *d_r, int d, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    res_r[i] = d_r[i + m] - d_r[i];
}

void fwht_nlogd(double* a, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a, fwht_r, N, k, m);
    if (flag) {
        for (int i = 0; i < N; i += k) fwht_parallel(fwht_r + i, k);
    } else {
        fwht_block<<<(N / k + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(fwht_r, N, k);
    }
    fwht_nlogd_compute<<<(d * m + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_r, fwht_r, d, m, r, k, bit_cnt);
    thrust::inclusive_scan(d_r, d_r + d * m, d_r);
    fwht_nlogd_store<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a, d_r, d, m);
}

__device__ void fft(Complex *a_c, int N, const Complex *w_c, const int *bit_rev) {
    for (int i = 0; i < N; ++i) {
        int j = bit_rev[i];
        if (i < j) std::swap(a_c[i], a_c[j]);
    }
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
        for (int i = 0; i < N; i += m) {
            const Complex *o = w_c;
            for (int j = i; j < i + gap; ++j, o += step) {
                Complex u = a_c[j], v = *o * a_c[j + gap];
                a_c[j] = u + v;
                a_c[j + gap] = u - v;
            }
        }
    }
}

__global__ void fft_bit_rev(const Complex *a_gpu, Complex *b_gpu, int N, const int *bit_rev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = bit_rev[i];
    b_gpu[j] = a_gpu[i];
}

__global__ void fft_butterfly(Complex *a_c, int gap, int step, const Complex *w_c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = (i / gap) * gap * 2, q = i % gap;
    int x = p + q, y = p + q + gap;
    Complex u = a_c[x], v = w_c[q * step] * a_c[y];
    a_c[x] = u + v;
    a_c[y] = u - v;
}

__global__ void fft_block(Complex *a_c, int N, const Complex *w_c, const int *bit_rev, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i * k >= N) return;
    fft(a_c + i * k, k, w_c, bit_rev);
}

void fft_parallel(Complex *a_c, int N, const Complex *w_c, const int *bit_rev) {
    if (N <= 2 * NUM_THREADS) {
        fft_block<<<1, 1>>>(a_c, N, w_c, bit_rev, N);
        return;
    }
    fft_bit_rev<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, b_c, N, bit_rev);
    cudaMemcpy(a_c, b_c, N * sizeof(Complex), cudaMemcpyDeviceToDevice);
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
        fft_butterfly<<<(N / NUM_THREADS) / 2, NUM_THREADS>>>(a_c, gap, step, w_c);
    }
}

__global__ void dft_nlogd_compute(Complex *res_c, Complex *dft_c, Complex *w_c, int d, int m, const int *r, int N, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d * m) return;
    int it = i / m, p = i % m;
    int j = r[it];
    int x = ((int64_t)j * p) % N, y = j % k;
    Complex v = dft_c[p * k + y];
    res_c[i] = v * w_c[x];
}

__global__ void dft_nlogd_store(Complex *res_c, Complex *d_c, int d, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    res_c[i] = d_c[i + m] - d_c[i];
}

void dft_nlogd(Complex* a_c, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dft_c, N, k, m);
    if (flag) {
        for (int i = 0; i < N; i += k) fft_parallel(dft_c + i, k, kw_c, kbit_rev);
    } else {
        fft_block<<<(N / k + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dft_c, N, kw_c, kbit_rev, k);
    }
    dft_nlogd_compute<<<(d * m + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_c, dft_c, w_c, d, m, r, N, k);
    thrust::inclusive_scan(d_c, d_c + d * m, d_c);
    dft_nlogd_store<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, d_c, d, m);
}

void dft(Complex *srft_c, int N) {
    fft_parallel(srft_c, N, w_c, bit_rev);
}

__global__ void dct_store(Complex *dct_c, Complex *a_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dct_c[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a_c[i];
}

__global__ void dct_load(Complex *a_c, Complex *dct_c, Complex *dct_shift_c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    a_c[i] = (dct_c[i] * dct_shift_c[i]).real();
}

void dct(Complex *a_c, int N) {
    dct_store<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dct_c, a_c, N);
    fft_parallel(dct_c, N, w_c, bit_rev);
    dct_load<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dct_c, dct_shift_c, N);
}

__global__ void dct_nlogd_load(Complex *a_c, Complex *dct_c, Complex *dct_shift_c, int d, const int *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    int j = r[i];
    a_c[i] = (dct_c[i] * dct_shift_c[j]).real();
}

void dct_nlogd(Complex *a, int N, int k, int d, const int *r) {
    dct_store<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(dct_c, a_c, N);
    dft_nlogd(dct_c, N, k, d, r);
    dct_nlogd_load<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(a_c, dct_c, dct_shift_c, d, r);
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
    cudaMalloc((void**) &srft_c, N * sizeof(Complex));
    cudaMalloc((void**) &f_gpu, N * sizeof(int));
    cudaMalloc((void**) &perm_gpu, N * sizeof(int));
    cudaMalloc((void**) &r_gpu, d * sizeof(int));
    cudaMemcpy(f_gpu, f, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(perm_gpu, perm, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(r_gpu, r, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &a_gpu, N * sizeof(double));
    cudaMalloc((void**) &sa_re_gpu, d * sizeof(double));
    cudaMalloc((void**) &sa_im_gpu, d * sizeof(double));
    if (transform == Transform::walsh) {
        cudaMalloc((void**) &srft_r, N * sizeof(double));
    }
    if (transform == Transform::fourier || transform == Transform::cosine) {
        cudaMalloc((void**) &b_c, N * sizeof(Complex));
        cudaMalloc((void**) &dft_c, N * sizeof(Complex));
        cudaMalloc((void**) &w_c, (N + 1) * sizeof(Complex));
        cudaMalloc((void**) &bit_rev, N * sizeof(int));
        compute_w<<<(N + NUM_THREADS) / NUM_THREADS, NUM_THREADS>>>(w_c, N);
        cudaMemcpy(w_c + N, w_c, sizeof(Complex), cudaMemcpyDeviceToDevice);
        compute_bit_rev<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(bit_rev, N);
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
        cudaMalloc((void**) &kw_c, (k + 1) * sizeof(Complex));
        cudaMalloc((void**) &kbit_rev, k * sizeof(int));
        cudaMalloc((void**) &d_c, N * sizeof(Complex));
        compute_w<<<(k + NUM_THREADS) / NUM_THREADS, NUM_THREADS>>>(kw_c, k);
        compute_bit_rev<<<(k + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(kbit_rev, k);
    }
    if (transform == Transform::walsh) {
        cudaMalloc((void**) &bit_cnt, N * sizeof(int));
        cudaMalloc((void**) &fwht_r, N * sizeof(double));
        cudaMalloc((void**) &d_r, N * sizeof(double));
        compute_bitcount<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(bit_cnt, N);
    }
    //flag = (N < (double)k * k * (31 - __builtin_clz(k)));
    flag = false;
}

__global__ void shuffle(Complex *srft_c, double *a_gpu, int *perm_gpu, int *f_gpu, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    srft_c[i] = Complex(a_gpu[perm_gpu[i]] * f_gpu[i], 0);
}

__global__ void shuffle_real(double *srft_r, double *a_gpu, int *perm_gpu, int *f_gpu, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    srft_r[i] = a_gpu[perm_gpu[i]] * f_gpu[i];
}

__global__ void srft_save(double *sa_re_gpu, double *sa_im_gpu, double scale, Complex *srft_c, int d, const int *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    sa_re_gpu[i] = srft_c[r[i]].real();
    sa_im_gpu[i] = srft_c[r[i]].imag();
}

__global__ void srft_save(double *sa_re_gpu, double *sa_im_gpu, double scale, Complex *srft_c, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    sa_re_gpu[i] = srft_c[i].real();
    sa_im_gpu[i] = srft_c[i].imag();
}

__global__ void srft_real_save(double *sa_re_gpu, double scale, double *srft_r, int d, const int *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    sa_re_gpu[i] = srft_r[r[i]];
}

void srft(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    cudaMemcpy(a_gpu, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (transform == Transform::walsh) {
        shuffle_real<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_r, a_gpu, perm_gpu, f_gpu, N);
        fwht_parallel(srft_r, N);
        cudaMemcpy(sa_re, srft_r, d * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        shuffle<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_c, a_gpu, perm_gpu, f_gpu, N);
        if (transform == Transform::fourier) {
            dft(srft_c, N);
        } else {
            assert(transform == Transform::cosine);
            dct(srft_c, N);
        }
        double scale = sqrt((double)N / d);
        srft_save<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(sa_re_gpu, sa_im_gpu, scale, srft_c, d, r);
        cudaMemcpy(sa_re, sa_re_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sa_im, sa_im_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

void srft_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
    cudaMemcpy(a_gpu, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (transform == Transform::walsh) {
        shuffle_real<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_r, a_gpu, perm_gpu, f_gpu, N);
        fwht_nlogd(srft_r, N);
        cudaMemcpy(sa_re, srft_r, d * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        shuffle<<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(srft_c, a_gpu, perm_gpu, f_gpu, N);
        if (transform == Transform::fourier) {
            dft(srft_c, N);
        } else {
            assert(transform == Transform::cosine);
            dct(srft_c, N);
        }
        double scale = sqrt((double)N / d);
        srft_save<<<(d + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(sa_re_gpu, sa_im_gpu, scale, srft_c, d);
        cudaMemcpy(sa_re, sa_re_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sa_im, sa_im_gpu, d * sizeof(double), cudaMemcpyDeviceToHost);
    }
}
