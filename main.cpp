#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// make random double array
void rand_double_array(double *a, const int n, const int seed) {
    std::srand(seed);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<double>(std::rand());
    }
}

void rand_complex_array(Complex *a, const int n, const int seed) {
    std::srand(seed);
    for (int i = 0; i < n; ++i) {
        double d1 = static_cast<double>(std::rand());
        double d2 = static_cast<double>(std::rand());
        a[i] = Complex(d1, d2);
    }
}


void rand_ComplexArr(ComplexArr &a, const int seed) {
    std::srand(seed);
    for (int i = 0; i < a.n; ++i) {
        double d1 = static_cast<double>(std::rand());
        double d2 = static_cast<double>(std::rand());
        CA_SET(a, d1, d2, i);
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: log2 of array size" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        std::cout << "-t <str>: the transform: one of {fwht, dft, idft}" << std::endl;
        std::cout << "-d <str>: number of elements to sample. must be <= n" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    // std::ofstream fsave(savename);

    char* ttype = find_string_option(argc, argv, "-t", nullptr);
    if (ttype == nullptr) {
        std::cout << "You must enter transform type!\n";
        exit(-1);
    }
    // Initialize
    // int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    int n = find_int_arg(argc, argv, "-n", 3);
    int d = find_int_arg(argc, argv, "-n", 2);
    if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwt")) n = 1 << n;
    // Complex *arr = new Complex[n];
    // rand_complex_array(arr, n, part_seed);
    ComplexArr arr(n);
    rand_ComplexArr(arr, part_seed);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

//     init_simulation(parts, num_parts, size);

// #ifdef _OPENMP
// #pragma omp parallel default(shared)
// #endif
//     {
//         for (int step = 0; step < nsteps; ++step) {
//             simulate_one_step(parts, num_parts, size);

//             // Save state if necessary
// #ifdef _OPENMP
// #pragma omp master
// #endif
//             if (fsave.good() && (step % savefreq) == 0) {
//                 save(fsave, parts, num_parts, size);
//             }
//         }
//     }
    if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwt")) {
        fwht(arr);
    } else if (!strcmp(ttype, "dft")) {
        // dft(arr, n);
    } else if (!strcmp(ttype, "idft")) {
        // idft(arr, n);
    } else {
        std::cout << "Not a supported transform type!\n";
        // delete[] arr;
        exit(-1);
    }
    

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    
    // change to output to file if -o set
    if (savename != nullptr && n < 33) {
        std::cout << "arr: ";
        for (int i = 0; i < n; ++i) {
            std::cout << "(" << arr[i].first << " " << arr[i].second << ")";
        }
        std::cout << "\n";
    }
    

    // delete[] arr;
    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for arr of size " << n << ".\n";
    // fsave.close();
}
