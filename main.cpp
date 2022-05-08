#include "common.h"
#include <chrono>
#include <fstream>
#include <iomanip>

// =================
// Helper Functions
// =================

// make random double array
std::mt19937 gen;

void rand_double_array(double* a, const int n, const int N) {
    std::fill(a, a + N, 0.0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < n; ++i) a[i] = dist(gen);
}

void rand_sign_array(int* f, const int N) {
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < N; ++i) f[i] = dist(gen) ? -1 : 1;
}

void rand_permutation(int* p, const int N) {
    for (int i = 0; i < N; ++i) p[i] = i;
    std::shuffle(p, p + N, gen);
}

void rand_subsample(int* p, const int N, const int subsample) {
    int q = new int [N];
    rand_permutation(q, N);
    std::copy(q, q + subsample, p);
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
//std::cout << "-n <int>: log2 of array size" << std::endl;
        std::cout << "-n <int>: array size" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set random seed" << std::endl;
        std::cout << "-r <int>: rank" << std::endl;
        std::cout << "-t <str>: the transform: one of {fwht, dft, dct}" << std::endl;
        std::cout << "-d <str>: number of elements to sample. must be <= N" << std::endl;
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
    int seed = find_int_arg(argc, argv, "-s", 0);
    int n_ranks = find_int_arg(argc, argv, "-r", 0);
    int n = find_int_arg(argc, argv, "-n", 3);
    int d = find_int_arg(argc, argv, "-d", 2);
    int N = 1;
    while (N <= n) N *= 2;
    assert(d <= n);
    gen.seed(seed);
    // if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwt"))
    //n = 1 << n;
    //d = 1 << d;
    // Complex *arr = new Complex[n];
    // rand_complex_array(arr, n, part_seed);

    double* input = new double[N]; rand_double_array(input, n, N);
    double* output_re = new double[d], *output_im = new double[d];
    int* perm = new int[N]; rand_permutation(perm, N);
    int* subsample = new int[d]; rand_subsample(subsample, N, d);
    int* flip = new int[N]; rand_sign_array(flip, N);

    // if (savename != nullptr && d < 33) {
    //     std::cout << "perm: ";
    //     for (int i = 0; i < d; ++i) {
    //         std::cout << perm[i] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

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
        srft(N, d, n_ranks, flip, perm, input, output_re, output_im, subsample, Transform::walsh);
    } else if (!strcmp(ttype, "dft") || !strcmp(ttype, "fft")) {
        srft(N, d, n_ranks, flip, perm, input, output_re, output_im, subsample, Transform::fourier);
    } else if (!strcmp(ttype, "dfts") || !strcmp(ttype, "ffts")) {
        srft_nlogd(N, d, n_ranks, flip, perm, input, output_re, output_im, subsample, Transform::walsh);
    } else if (!strcmp(ttype, "fwhts") || !strcmp(ttype, "fwts")) {
        srft_nlogd(N, d, n_ranks, flip, perm, input, output_re, output_im, subsample, Transform::fourier);
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
    
    delete[] a;
    delete[] sa;
    delete[] space;
    delete[] perm;
    delete[] r;
    delete[] f;
    // Finalize
    std::cout << std::fixed << "Simulation Time = " << seconds << " seconds for arr of size " << n << " using transform " << ttype << " with seed " << s << " and d " << d << " and rank " << rank << ".\n";
    // fsave.close();
}
