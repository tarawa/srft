#include "common.h"

double *srft_re, *srft_im, *w_re, *w_im, *kw_re, *kw_im, *dft_re, *dft_im, *dct_re, *dct_im, *fwht_re;
int *bit_cnt, *bit_rev, *kbit_rev, k;
bool flag;

void transpose(const double *a, double *temp, int N, int k) {
    int m = N / k;
#pragma omp for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            temp[i * k + j] = a[j * m + i];
}

void compute_w(double* w_re, double* w_im, int N) {
#pragma omp parallel for
    for (int i = 0; i <= N; ++i) w_re[i] = cos(2. * PI * i / N), w_im[i] = sin(2. * PI * i / N);
}

void compute_bit_rev(int* bit_rev, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int j = 0;
        for (int x = (N >> 1), y = i; x; x >>= 1, y >>= 1) j = ((j << 1) | (y & 1));
        bit_rev[i] = j;
    }
}

void compute_bitcount(int *bit_cnt, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int temp = (i & 0x55555555) + ((i >> 1) & 0x55555555);
        temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
        temp = (temp & 0x0F0F0F0F) + ((temp >> 4) & 0x0F0F0F0F);
        temp = (temp & 0x00FF00FF) + ((temp >> 8) & 0x00FF00FF);
        temp = (temp & 0x0000FFFF) + ((temp >> 16) & 0x0000FFFF);
        bit_cnt[i] = temp;
    }
}

void fwht(double* a, int N) {
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                double x = a[j];
                double y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
    }
}

void fwht_parallel(double* a, int N) {
    for (int h = 1; h < N; h *= 2) {
#pragma omp for collapse(2)
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

void fwht_nlogd(double* a, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose(a, fwht_re, N, k);
    if (flag) {
        for (int i = 0; i < N; i += k) fwht_parallel(fwht_re + i, k);
    } else {
#pragma omp for
        for (int i = 0; i < N; i += k) fwht(fwht_re + i, k);
    }
#pragma omp for collapse(2)
    for (int i = 0; i < d; ++i) {
        for (int p = 0; p < m; ++p) {
            int j = r[i], y = j / m;
            // m is a power of 2
            // <j, p | q * m> = <j, p> + <j, q * m> = <j, p> + <j / m, q>
            // sum_q -1^{<j, p | q * m>} = sum_q -1^{<j, p>} -1^{<j / m, q>}
            // what we have: sum_q -1^{y * q} for each y
            double v = fwht_re[p * k + y];
            double x = (bit_cnt[j & p] ? -v : v);
#pragma omp atomic
            a[i] += x;
        }
    }
}

void fft(double* a_re, double *a_im, int N, const double *w_re, const double *w_im, const int *bit_rev) {
    for (int i = 0; i < N; ++i) {
        int j = bit_rev[i];
        if (i < j) {
            std::swap(a_re[i], a_re[j]);
            std::swap(a_im[i], a_im[j]);
        }
    }
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
        for (int i = 0; i < N; i += m) {
            const double *o_re = w_re, *o_im = w_im;
            for (int j = i; j < i + gap; ++j, o_re += step, o_im += step) {
                double u_re = a_re[j], u_im = a_im[j];
                double v_re = *o_re * a_re[j + gap] - *o_im * a_im[j + gap];
                double v_im = *o_re * a_im[j + gap] + *o_im * a_re[j + gap];
                a_re[j] = u_re + v_re;
                a_im[j] = u_im + v_im;
                a_re[j + gap] = u_re - v_re;
                a_im[j + gap] = u_im - v_im;
            }
        }
    }
}

void fft_parallel(double* a_re, double *a_im, int N, const double *w_re, const double *w_im, const int *bit_rev) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        int j = bit_rev[i];
        if (i < j) {
            std::swap(a_re[i], a_re[j]);
            std::swap(a_im[i], a_im[j]);
        }
    }
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
#pragma omp for collapse(2)
        for (int i = 0; i < N; i += m) {
            for (int j = 0; j < gap; ++j) {
                double u_re = a_re[i + j], u_im = a_im[j];
                double v_re = w_re[j * step] * a_re[i + j + gap] - w_im[j * step] * a_im[i + j + gap];
                double v_im = w_re[j * step] * a_im[i + j + gap] + w_im[j * step] * a_re[i + j + gap];
                a_re[i + j] = u_re + v_re;
                a_im[i + j] = u_im + v_im;
                a_re[i + j + gap] = u_re - v_re;
                a_im[i + j + gap] = u_im - v_im;
            }
        }
    }
}

void dft_nlogd(double* a_re, double *a_im, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose(a_re, dft_re, N, k);
    transpose(a_im, dft_im, N, k);
    if (flag) {
        for (int i = 0; i < N; i += k) fft_parallel(dft_re + i, dft_im + i, k, kw_re, kw_im, kbit_rev);
    } else {
#pragma omp for
        for (int i = 0; i < N; i += k) fft(dft_re + i, dft_im + i, k, kw_re, kw_im, kbit_rev);
    }
#pragma omp for
    for (int i = 0; i < d; ++i) a_re[i] = a_im[i] = 0;
#pragma omp for collapse(2)
    for (int i = 0; i < d; ++i) {
        for (int p = 0; p < m; ++p) {
            int j = r[i];
            int y = j % k;
            // j * (p + q * m) / N = j * p / N + j * q / k
            // sum_q e^{2pi I j * (p + q * m) / N} = sum_q e^{2pi I (j * p / N) + (j * q / k)}
            // what we have: sum_q e^{2pi I y * q / k} for each y
            int x = ((int64_t)j * p) % N;
            double v_re = dft_re[p * k + y], v_im = dft_im[p * k + y];
            double z_re = w_re[x], z_im = w_im[x];
#pragma omp atomic
            a_re[i] += v_re * z_re - v_im * z_im;
#pragma omp atomic
            a_im[i] += v_re * z_im + v_im * z_re;
        }
    }
}

void dft(double *a_re, double *a_im, int N) {
    fft_parallel(a_re, a_im, N, w_re, w_im, bit_rev);
}

double *dct_x, *dct_y;

void dct(double *a, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        dct_re[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i];
        dct_im[i] = 0.;
    }
    fft_parallel(dct_re, dct_im, N, w_re, w_im, bit_rev);
#pragma omp for
    for (int i = 0; i < N; ++i) {
        a[i] = dct_x[i] * dct_re[i] - dct_y[i] * dct_im[i];
    }
}


void dct_nlogd(double *a, int N, int k, int d, const int *r) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        dct_re[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i];
        dct_im[i] = 0.;
    }
    dft_nlogd(dct_re, dct_im, N, k, d, r);
#pragma omp for
    for (int i = 0; i < d; ++i) {
        int j = r[i];
        a[i] = dct_x[j] * dct_re[i] - dct_y[j] * dct_im[i];
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
    srft_re = new double[N];
    srft_im = new double[N];
    if (transform == Transform::fourier || transform == Transform::cosine) {
        dft_re = new double[N];
        dft_im = new double[N];
        w_re = new double[N + 1];
        w_im = new double[N + 1];
        bit_rev = new int[N];
        compute_w(w_re, w_im, N);
        compute_bit_rev(bit_rev, N);
    }
    if (transform == Transform::cosine) {
        dct_re = new double[N];
        dct_im = new double[N];
        dct_x = new double[N + 1];
        dct_y = new double[N + 1];
#pragma omp parallel for
        for (int i = 0; i <= N; ++i) {
            dct_x[i] = 2 * cos(PI * i / 2. / N);
            dct_y[i] = 2 * sin(PI * i / 2. / N);
        }
    }
}

void init_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform) {
    k = 2;
    for (int i = 1; k < d * i && k < N; ++i) k *= 2;
    init(N, d, n_ranks, f, perm, r, transform);
    if (transform == Transform::fourier || transform == Transform::cosine) {
        kw_re = new double[k + 1];
        kw_im = new double[k + 1];
        kbit_rev = new int[k];
        compute_w(kw_re, kw_im, k);
        compute_bit_rev(kbit_rev, k);
    }
    if (transform == Transform::walsh) {
        fwht_re = new double[N];
        bit_cnt = new int[N];
        compute_bitcount(bit_cnt, N);
    }
    //flag = (N < (double)k * k * (31 - __builtin_clz(k)));
    flag = false;
}

void srft(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        srft_re[i] = a[perm[i]] * f[i];
        srft_im[i] = 0.;
    }
    if (transform == Transform::walsh) {
        fwht_parallel(srft_re, N);
    } else if (transform == Transform::fourier) {
        dft(srft_re, srft_im, N);
    } else {
        assert(transform == Transform::cosine);
        dct(srft_re, N);
    }
    double scale = sqrt(N / d);
#pragma omp for
    for (int i = 0; i < d; ++i) {
        sa_re[i] = srft_re[r[i]] * scale;
        sa_im[i] = srft_im[r[i]] * scale;
    }
}

void srft_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const double *a, double *sa_re, double *sa_im, const int *r, Transform transform) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        srft_re[i] = a[perm[i]] * f[i];
        srft_im[i] = 0.;
    }
    if (transform == Transform::walsh) {
        fwht_nlogd(srft_re, N, k, d, r);
    } else if (transform == Transform::fourier) {
        dft_nlogd(srft_re, srft_im, N, k, d, r);
    } else {
        assert(transform == Transform::cosine);
        dct_nlogd(srft_re, N, k, d, r);
    }
    double scale = sqrt(N / d);
#pragma omp for
    for (int i = 0; i < d; ++i) {
        sa_re[i] = srft_re[i] * scale;
        sa_im[i] = srft_im[i] * scale;
    }
}
