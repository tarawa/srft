#include "common.h"

// make sure compilation works, delete later
void do_stuff(void) {
    std::cout << "serial stuff\n";
}

void fwht(double* a, const int n) {
    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                double x = a[j];
                double y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

void fwht_subsampled(double* v, int n, int d, int* perm, double* sa) {

}

/* 
* @params n: array size
* @params d: sample size
* @params r: d random indices
* @params f: vector of n random signs (-1 or +1)
* @params perm: permutation vector, to permute a
* @params a: array to be srft'd
* @params space: array pre-sampling
* @params sa: destination to store result in
* @params transform: the transformation function
*/
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

void dft(double* a, int N) {
    double w[N];
    fft(w, N);
    for (int i = 0, j = 0; i < N; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = N >> 1; (j ^= k) < k;) k >>= 1;
    }
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
        for (int i = 0; i < N; i += m) {
            double *o = w;
            for (int j = i; j < i + gap; ++j, o += step) {
            double u = a[j], v = *o * a[j + gap];
            a[j] = u + v;
            a[j + gap] = u - v;
            }
        }
    }
}

void idft(double* a, int N) {
    dft(a, N);
    std::reverse(a + 1, a + N);
    for (int i = 0; i < N; ++i) a[i] /= N;
}