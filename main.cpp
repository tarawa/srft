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
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    // std::ofstream fsave(savename);

    // Initialize
    // int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    int n = 1<<find_int_arg(argc, argv, "-n", 3);
    double * arr = new double[n];
    rand_double_array(arr, n, part_seed);

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
    fwht(arr, n);

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    
    // change to output to file if -o set
    if (savename != nullptr && n < 33) {
        std::cout << "arr: ";
        for (int i = 0; i < n; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }
    

    delete[] arr;
    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for arr of size " << n << ".\n";
    // fsave.close();
}
