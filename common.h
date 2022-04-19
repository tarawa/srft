#include <complex>
#include <cmath>
#include <iostream>
#include <complex>
#include <algorithm>

// #define Complex std::complex<double>
#define PI M_PI

class Complex {
    public:
        double real_part;
        double imaginary_part;

    Complex() {}

    Complex(double r, double i)
        : real_part(r), imaginary_part(i)
        {}

    friend std::ostream& operator<<(std::ostream &s, const Complex &c) {
        return s << "(" << c.real_part << ", " << c.imaginary_part << ")";
    }

    inline Complex operator+(const Complex& other) {
        return Complex(real_part + other.real_part, imaginary_part + other.imaginary_part);
    }

    inline Complex operator-(const Complex& other) {
        return Complex(real_part - other.real_part, imaginary_part - other.imaginary_part);
    }

    inline void operator/=(const int& other) {
        real_part /= other;
        imaginary_part /= other;
    }

    inline Complex operator*(const Complex& other) {
        return Complex(
            real_part * other.real_part - imaginary_part * other.imaginary_part,
            real_part * other.imaginary_part + imaginary_part * other.real_part
        );
    }

    inline void operator+=(const Complex& other) {
        real_part += other.real_part;
        imaginary_part += other.imaginary_part;
    }    

};

void do_stuff(void);

void fwht(Complex a[], const int n);

void dft(Complex a[], int N);

void fft(Complex w[], int N);

void idft(Complex a[], int N);

