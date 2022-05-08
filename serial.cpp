#include "common.h"

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

void compute_bitcount(int *bit_cnt, int N) {
    for (int i = 0; i < N; ++i) {
        int temp = (i & 0x55555555) + ((i >> 1) & 0x55555555);
        temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
        temp = (temp & 0x0F0F0F0F) + ((temp >> 4) & 0x0F0F0F0F);
        temp = (temp & 0x00FF00FF) + ((temp >> 8) & 0x00FF00FF);
        temp = (temp & 0x0000FFFF) + ((temp >> 16) & 0x0000FFFF);
        bit_cnt[i] = temp;
    }
}

void fwht_nlogd(double* a, int N, int k, int d, const int *subsample) {
    int m = N / k;
    double *temp = new double[N];
    transpose(a, temp, N, k);
    for (int i = 0; i < N; i += k) fwht(temp + i, k);
    int *bit_cnt = new int[N];
    compute_bitcount(bit_cnt, N);
    for (int i = 0; i < d; ++i) {
        int j = subsample[i], y = j / m;
        // m is a power of 2
        // <j, p | q * m> = <j, p> + <j, q * m> = <j, p> + <j / m, q>
        // sum_q -1^{<j, p | q * m>} = sum_q -1^{<j, p>} -1^{<j / m, q>}
        // what we have: sum_q -1^{y * q} for each y
        double u = 0;
        for (int p = 0; p < m; ++p) {
            int x = (bit_cnt[j & p] ? -1 : 1);
            double v = temp[p * k + y];
            u += x * v;
        }
        a[i] = u;
    }
}

void compute_w(double* w_re, double* w_im, int N) {
    for (int i = 0; i <= N; ++i) w_re[i] = cos(2. * PI * i / N), w_im[i] = sin(2. * PI * i / N);
}

void compute_bit_rev(int* bit_rev, int N) {
    for (int i = 0; i < N; ++i) {
        int j = 0;
        for (int x = N, y = i; x; x >>= 1, y >>= 1) j = ((j << 1) | (y & 1));
        bit_rev[i] = j;
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

void transpose(const double *input, double *output, int N, int k) {
    int m = N / k;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            output[i * k + j] = input[j * m + i];
}

void dft_nlogd(double* a_re, double *a_im, int N, int k, int d, const int *subsample) {
    double *temp_re = new double[N], *temp_im = new double[N];
    int m = N / k;
    transpose(a_re, temp_re, N, k);
    transpose(a_im, temp_im, N, k);
    double *w_re = new double[k + 1], *w_im = new double[k + 1];
    int *bit_rev = new int[k];
    compute_w(w_re, w_im, k);
    compute_bit_rev(bit_rev, k);
    for (int i = 0; i < N; i += k) fft(temp_re + i, temp_im + i, k, w_re, w_im, bit_rev);
    delete[] w_re;
    delete[] w_im;
    w_re = new double[N + 1], w_im = new double[N + 1];
    compute_w(w_re, w_im, N);
    for (int i = 0; i < d; ++i) {
        int j = subsample[i];
        int y = j % k;
        // j * (p + q * m) / N = j * p / N + j * q / k
        // sum_q e^{2pi I j * (p + q * m) / N} = sum_q e^{2pi I (j * p / N) + (j * q / k)}
        // what we have: sum_q e^{2pi I y * q / k} for each y
        double u_re = 0, u_im = 0;
        for (int p = 0; p < m; ++p) {
            int x = ((int64_t)j * p) % N;
            double v_re = temp_re[p * k + y], v_im = temp_im[p * k + y];
            double z_re = w_re[x], z_im = w_im[x];
            u_re += v_re * z_re - v_im * z_im;
            u_im += v_re * z_im + v_im * z_re;
        }
        a_re[i] = u_re;
        a_im[i] = u_im;
    }
}

/* 
* @params N: array size
* @params d: subsample size
* @params n_ranks: num_ranks
* @params flip: vector of N random signs (-1 or +1)
* @params perm: random permutation of [0, N)
* @params input: array to be srft'd
* @params subsample: d random elements from [0, N) (to be subsampled)
* @params output_re: destination to store real part of result
* @params output_im: destination to store imaginary part of result
* @params transform: the transformation to be performed
*/

void dft(double *a_re, double *a_im, int N) {
    double *w_re = new double[N + 1], *w_im = new double[N + 1];
    int *bit_rev = new int[N];
    compute_w(w_re, w_im, N);
    compute_bit_rev(bit_rev, N);
    fft(temp_re, temp_im, N, w_re, w_im, bit_rev);
}

void dct(double *a, int N) {
    double *temp_re = new double[N], *temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i], temp_im[i] = 0.;
    double *w_re = new double[N + 1], *w_im = new double[N + 1];
    int *bit_rev = new int[N];
    compute_w(w_re, w_im, N);
    compute_bit_rev(bit_rev, N);
    fft(temp_re, temp_im, N, w_re, w_im, bit_rev);
    for (int i = 0; i < N; ++i) {
        double x = 2 * cos(PI * i / 2 / N), y = 2 * sin(PI * i / 2 / N);
        a[i] = x * temp_re[i] - y * temp_im[i];
    }
}

void dct_nlogd(double *a, int N, int k, int d, const int *subsample) {
    double *temp_re = new double[N], *temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i], temp_im[i] = 0.;
    dft_nlogd(temp_re, temp_im, N, k, d, subsample);
    for (int i = 0; i < d; ++i) {
        double x = 2 * cos(PI * i / 2 / N), y = 2 * sin(PI * i / 2 / N);
        a[i] = x * temp_re[i] - y * temp_im[i];
    }
}

void srft(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform) {
    double *temp_re = new double[N], temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[i] = input[perm[i]] * flip[i], temp_im[i] = 0.;
    if (transform == Transform::walsh) {
        fwht(temp_re, N);
    } else if (transform == Transform::fourier) {
        dft(temp_re, temp_im, N)
    } else {
        assert(transform == Transform::cosine);
        dct(temp_re, N);
    }
    double scale = sqrt(N / d);
    for (int i = 0; i < d; ++i) output_re[i] = temp_re[subsample[i]] * scale, output_im[i] = temp_im[subsample[i]] * scale;
}

void srft_nlogd(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform) {
    double *temp_re = new double[N], temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[i] = input[perm[i]] * flip[i], temp_im[i] = 0.;
    int k = 2;
    for (int i = 1; k < d * i; ++i) k *= 2;
    if (transform == Transform::walsh) {
        fwt_nlogd(temp_re, temp_im, N, k, d, subsample);
    } else if (transform == Transform::fourier) {
        dft_nlogd(temp_re, temp_im, N, k, d, subsample);
    } else {
        assert(transform == Transform::cosine);
        dct_nlogd(temp_re, temp_im, N, k, d, subsample);
    }
    double scale = sqrt(N / d);
    for (int i = 0; i < d; ++i) {
        output_re[i] = temp_re[i] * scale;
        output_im[i] = temp_im[i] * scale;
    }
}
