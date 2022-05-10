#include "common.h"

typedef std::complex<double> Complex;

Complex *srft_c, *w_c, *kw_c, *dft_c, *dct_c, *fwht_c, *dct_shift_c;
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

void compute_w(Complex *w_c, int N) {
#pragma omp parallel for
    for (int i = 0; i <= N; ++i) w_c[i] = Complex(cos(2. * PI * i / N), sin(2. * M_PI * i / N));
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

void fwht_block(double* a, int N, int k) {
    for (int h = 1; h < k; h *= 2) {
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
    fwht_block(fwht_re, N, k);
#pragma omp for
    for (int i = 0; i < d; ++i) a[i] = 0;
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

void fft_parallel(Complex *a_c, int N, const Complex *w_c, const int *bit_rev, int k) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        int t = (i & (k - 1));
        int j = i - t + kbit_rev[t];
        if (i < j) {
            std::swap(a_c[i], a_c[j]);
        }
    }
    for (int m = 2; m <= k; m *= 2) {
        int gap = m / 2, step = k / m;
#pragma omp for collapse(2)
        for (int i = 0; i < N; i += m) {
            for (int j = 0; j < gap; ++j) {
                Complex u = a_c[i + j], v = w_c[j * step] * a_c[i + j + gap];
                a_c[i + j] = u + v;
                a_c[i + j + gap] = u - v;
            }
        }
    }
}

void dft_nlogd(Complex* a_c, int N, int k, int d, const int *r) {
    int m = N / k;
    transpose(a_c, dft_c, N, k);
    fft_parallel(dft_c, N, kw_c, kbit_rev, k);
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
    fft_parallel(a_c, N, w_c, bit_rev, N);
}

void dct(double *a, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        dct_c[(i & 1) ? N - 1 - (i >> 1) : (i >> 1)] = a[i];
    }
    fft_parallel(dct_c, N, w_c, bit_rev, N);
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
    if (transform == Transform::cosine || transform == Transform::walsh) {
        srft_re = new double[N];
    }
    if (transform == Transform::fourier || transform == Transform::cosine) {
        dft_c = new Complex[N];
        w_c = new Complex[N + 1];
        bit_rev = new int[N];
        compute_w(w_c, N);
        compute_bit_rev(bit_rev, N);
    }
    if (transform == Transform::cosine) {
        dct_c = new Complex[N];
        dct_shift_c = new Complex[N + 1];
#pragma omp parallel for
        for (int i = 0; i <= N; ++i) {
            dct_shift_c[i] = Complex(2 * cos(M_PI * i / 2. / N), 2 * sin(M_PI * i / 2. / N));
        }
    }
}

void init_nlogd(int N, int d, int n_ranks, const int *f, const int *perm, const int *r, Transform transform) {
    k = 2;
    for (int i = 1; k < d * i && k < N; ++i) k *= 2;
    init(N, d, n_ranks, f, perm, r, transform);
    b_re = new double[N];
    b_im = new double[N];
    if (transform == Transform::fourier || transform == Transform::cosine) {
        kw_c = new Complex[k + 1];
        kbit_rev = new int[k];
        compute_w(kw_c, k);
        compute_bit_rev(kbit_rev, k);
    }
    if (transform == Transform::walsh) {
        fwht_re = new double[N];
        bit_cnt = new int[N];
        compute_bitcount(bit_cnt, N);
    }
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
        fwht_parallel(srft_re, N);
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
        fwht_nlogd(srft_re, N, k, d, r);
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
