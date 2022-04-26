#include "common.h"

// make sure compilation works, delete later
void do_stuff(void) {
    std::cout << "serial stuff\n";
}

void fwht(double* a, const long long n) {
    long long h = 1;
    while (h < n) {
        for (long long i = 0; i < n; i += h * 2) {
            for (long long j = i; j < i + h; ++j) {
                double x = a[j];
                double y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
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
void srft(long long n, int d, int* r, int* f, int* perm, double* a, double* space, double* sa, void (*transform) (double*, long long)) {
    // apply perm and random signs
    if (a != space) {
        // not in place
        for (long long i = 0; i < n; ++i) {
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

void fft(double* w, long long N) {
    for (long long i = 0; i <= N; ++i) w[i] = cos(2. * PI * i / N);
}

void dft(double* a, long long N) {
    double w[N];
    fft(w, N);
    for (long long i = 0, j = 0; i < N; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (long long k = N >> 1; (j ^= k) < k;) k >>= 1;
    }
    for (long long m = 2; m <= N; m *= 2) {
        long long gap = m / 2, step = N / m;
        for (long long i = 0; i < N; i += m) {
            double *o = w;
            for (long long j = i; j < i + gap; ++j, o += step) {
            double u = a[j], v = *o * a[j + gap];
            a[j] = u + v;
            a[j + gap] = u - v;
            }
        }
    }
}

void idft(double* a, long long N) {
    dft(a, N);
    std::reverse(a + 1, a + N);
    for (long long i = 0; i < N; ++i) a[i] /= N;
}