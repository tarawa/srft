#include "common.h"

// make sure compilation works, delete later
void do_stuff(void) {
    std::cout << "serial stuff\n";
}

void fwht(Complex a[], const int n) {
    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                Complex x = a[j];
                Complex y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

void fwht(ComplexArr &a) {
    int h = 1;
    while (h < a.n) {
        for (int i = 0; i < a.n; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                std::pair<double, double> x = a[j];
                std::pair<double, double> y = a[j + h];
                a.set(x.first + y.first, x.second + y.second, j);
                a.set(x.first - y.first, x.second - y.second, j + h);
            }
        }
        h *= 2;
    }
}

void fft(Complex w[], int N) {
    for (int i = 0; i <= N; ++i) w[i] = Complex(cos(2. * PI * i / N), sin(2. * PI * i / N));
}

void dft(Complex a[], int N) {
    Complex w[N];
    fft(w, N);
    for (int i = 0, j = 0; i < N; ++i) {
        if (i < j) std::swap(a[i], a[j]);
        for (int k = N >> 1; (j ^= k) < k;) k >>= 1;
    }
    for (int m = 2; m <= N; m *= 2) {
        int gap = m / 2, step = N / m;
        for (int i = 0; i < N; i += m) {
            Complex *o = w;
            for (int j = i; j < i + gap; ++j, o += step) {
            Complex u = a[j], v = *o * a[j + gap];
            a[j] = u + v;
            a[j + gap] = u - v;
            }
        }
    }
}

void idft(Complex a[], int N) {
    dft(a, N);
    std::reverse(a + 1, a + N);
    for (int i = 0; i < N; ++i) a[i] /= N;
}