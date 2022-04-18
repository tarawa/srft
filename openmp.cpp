#include "common.h"
#include <iostream>

// make sure compilation works, delete later
void do_stuff(void) {
    std::cout << "omp stuff\n";
}

void fwht(double* a, const int n) {
    int h = 1;
    while (h < n) {
        #pragma omp parallel for
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