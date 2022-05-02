#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iomanip>

// =================
// Helper Functions
// =================

// make random double array

void rand_double_array(double* a, const int n, const int seed) {
    static std::default_random_engine gen(seed);
    static std::uniform_real_distribution<> dis(-1000.0, 1000.0);
    for (int i = 0; i < n; ++i)
        a[i] = dis(gen);
}

void rand_sign_array(int* f, const int n, const int seed) {
    std::srand(seed);
    for (int i = 0; i < n; ++i)
        f[i] = 1 - (2 * (rand() % 2));
}

void rand_permutation(int* p, const int n, const int d, const int seed) {
    std::srand(seed);
    int perm_start[n];
    for (int i = 0; i < n; ++i) {
        perm_start[i] = i;
    }
    std::random_shuffle(perm_start, perm_start + n - 1);
    for (int i = 0; i < d; ++i) {
        p[i] = perm_start[i];
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

long long find_ll_arg(int argc, char** argv, const char* option, long long default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoll(argv[iplace + 1]);
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
        std::cout << "-n <long long>: array size" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        std::cout << "-r <int>: rank" << std::endl;
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
    int s = find_int_arg(argc, argv, "-s", 0);
    int rank = find_int_arg(argc, argv, "-r", 0);
    long long n = find_ll_arg(argc, argv, "-n", 3);
    int d = find_int_arg(argc, argv, "-d", 2);
    if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwt")) n = 1 << n;
    // Complex *arr = new Complex[n];
    // rand_complex_array(arr, n, part_seed);


    double* a = new double[n]; rand_double_array(a, n, s);
    double* space = new double[n];
    double* sa = new double[d];
    int* perm = new int[n]; rand_permutation(perm, n, n, s);
    int* r = new int[d]; rand_permutation(r, n, d, s);
    int* f = new int[n]; rand_sign_array(f, n, s);
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
        // fwht(a, n);
        srft(n, d, r, f, perm, a, space, sa, fwht);
    } else if (!strcmp(ttype, "dft")) {
        // dft(a, n);
        srft(n, d, r, f, perm, a, space, sa, dft);
    } else if (!strcmp(ttype, "idft")) {
        // idft(a, n);
        srft(n, d, r, f, perm, a, space, sa, idft);
    } else {
        std::cout << "Not a supported transform type!\n";
        // delete[] a;
        exit(-1);
    }
    

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    
    // change to output to file if -o set
    if (savename != nullptr && d < 33) {
        std::cout << "arr: ";
        for (int i = 0; i < d; ++i) {
            std::cout << sa[i] << " ";
        }
        std::cout << "\n";
    }
    

    // delete[] arr;
    // Finalize
    std::cout << std::fixed << "Simulation Time = " << seconds << " seconds for arr of size " << n << " using transform " << ttype << " with seed " << s << " and d " << d << " and rank " << rank << ".\n";
    // fsave.close();
}
