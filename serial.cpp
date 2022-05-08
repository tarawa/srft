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

void compute_w(double* w_re, double* w_im, int N) {
    for (int i = 0; i <= N; ++i) w_re[i] = cos(2. * PI * i / N), w_im[i] = sin(2. * PI * i / N);
}

void fft(double* a_re, double *a_im, int N, const double *w_re, const double *w_im) {
    for (int i = 0, j = 0; i < N; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = N >> 1; (j ^= k) < k;) k >>= 1;
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

void fft_nlogd(double* a_re, double *a_im, int N, int k, int d, const int *subsample) {
    double *temp_re = new double[N], *temp_im = new double[N];
    int m = N / k;
    double *w_re = new double[k + 1], *w_im = new double[k + 1];
    transpose(a_re, temp_re, N, k);
    transpose(a_im, temp_im, N, k);
    for (int i = 0; i < N; i += k) fft(temp_re + i, temp_im + i, k, w_re, w_im);
    delete[] w_re;
    delete[] w_im;
    w_re = new double[N + 1], w_im = new double[N + 1];
    compute_w(w_re, w_im, N);
    for (int i = 0; i < d; ++i) {
        int j = subsample[i];
        int y = j % k;
        // j * (p + q * m) / N = j * p / N + j * q / k
        // sum_q e^{2pi I j * (p + q * m) / N} = sum_q e^{2pi I (j * p / N) + (j * q / k)}
        // what we have: sum_q e^{2pi I x * q / k} for each x
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

void srft(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform) {
    double *temp_re = new double[N], temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[i] = input[perm[i]] * flip[i], temp_im[i] = 0.;
    if (transform == Transform::walsh) {
        fwht(temp_re, N);
    } else if (transform == Transform::fourier) {
        double *w_re = new double[N + 1], *w_im = new double[N + 1];
        compute_w(w_re, w_im, N);
        fft(temp_re, temp_im, N, w_re, w_im);
    }
    for (int i = 0; i < d; ++i) output_re[i] = temp_re[subsample[i]], output_im[i] = temp_im[subsample[i]];
}

void srft_nlogd(int N, int d, int n_ranks, const int *flip, const int *perm, const double *input, double *output_re, double *output_im, const int *subsample, Transform transform) {
    double *temp_re = new double[N], temp_im = new double[N];
    for (int i = 0; i < N; ++i) temp_re[i] = input[perm[i]] * flip[i], temp_im[i] = 0.;
    int k = 2;
    for (int i = 1; k < d * i; ++i) k *= 2;
    if (transform == Transform::walsh) {
    //    fwht(temp_re, N);
    // TODO: FWT
    } else if (transform == Transform::fourier) {
        fft_nlogd(temp_re, temp_im, N, k, d, subsample);
    }
    for (int i = 0; i < d; ++i) {
        output_re[i] = temp_re[i];
        output_im[i] = temp_im[i];
    }
}

/*
void srft(int n, int d, int* r, int* f, int* perm, double* a, double* space, double* sa, void (*transform) (double*, int)) {
    // apply perm and random signs
    if (a != space) {
        // not in place
        for (int i = 0; i < n; ++i) {
            space[i] = a[perm[i]] * f[i];
        }
    }
    // apply transform
    transform(space, n);
    // sample and multiply by sqrt(n/d)
    double mult_fac = sqrt(static_cast<double>(d)/static_cast<double>(n));
    for (int i = 0; i < d; ++i) {
        sa[i] = mult_fac * space[r[i]];
    }
}

void srft_nlogd(int n, int d, int* r, int* f, int* perm, double* a, double* space, double* sa, void (*transform) (double*, int, int, int*, double*)) {
    if (a != space) {
        // not in place
        for (int i = 0; i < n; ++i) {
            space[i] = a[perm[i]] * f[i];
        }
    }

    transform(space, n, d, r, sa);

    double mult_fac = sqrt(static_cast<double>(d)/static_cast<double>(n));
    for (int i = 0; i < d; ++i) {
        sa[i] *= mult_fac;
    }
}

void fft(double* w, int N) {
    for (int i = 0; i <= N; ++i) w[i] = cos(2. * PI * i / N);
}

static inline void transpose(double* a, int width, int height) {
    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; i++)
            if (i < j)
                std::swap(a[i], a[j]);
}

//nlogd
//assumes n is divisible by d
// https://edoliberty.github.io/papers/approximationOfMatrices.pdf
void dft_subsampled(double* v, int n, int d, int* r, double* sa) {
    // Viewing the vector v as an l×m matrix V stored in row-major order, form the product W of the l×l unnormalized discrete Fourier transform F(l) and V , so that...
    int m = n/d;
    for (int i = 0; i < m; ++i)
        dft(v + (i * d), d);
    // Multiply the entry in row j and column k of W by e−2πi(j−1)(k−1)/n for j = 1, 2,...,l−1,l and k = 1, 2,...,m−1,m, in order to obtain the l x m matrix X
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < d; ++j)
            v[k*d + j] *= cos(2. * PI * j * k/n);
    // Transpose X to obtain an m × l matrix Y , so that...
    transpose(v, d, m);
    //  Form the product Z of the m × m unnormalized discrete Fourier transform F(m) and Y , so that
    // If we only need to compute l entries of z = F(n)v, then we can use Steps 1–3 above in their entirety to obtain Y ,and then compute the desired entries of z directly from the entries of Y
    double w[n];
    fft(w, n);
    for (int i = 0; i < d; ++i) {
        int index = r[i];
        sa[i] = dft_single(v, m, d, w, index);
    }
}

double dft_single(double* a, int m, int d, double* w, int index) {
    // TODO
    int i1 = index % m;
    int i2 = index / m;
    // std::cout << "\nindex: " << index << "\nm: " << m << "\nd: " << d << "\n";
    dft(a + i2 * m, m);
    return a[i2 * m + i1];
}


void idft(double* a, int N) {
    dft(a, N);
    std::reverse(a + 1, a + N);
    for (int i = 0; i < N; ++i) a[i] /= N;
}
 */