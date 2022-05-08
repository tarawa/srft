#include "common.h"
#include <chrono>
#include <fstream>
#include <iomanip>

// =================
// Helper Functions
// =================

// make random double array
std::mt19937 gen;

void rand_double_array(double *a, const int n, const int N) {
    std::fill(a, a + N, 0.0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < n; ++i) a[i] = dist(gen);
}

void rand_sign_array(int *f, const int N) {
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < N; ++i) f[i] = dist(gen) ? -1 : 1;
}

void rand_permutation(int *p, const int N) {
    for (int i = 0; i < N; ++i) p[i] = i;
    std::shuffle(p, p + N, gen);
}

void rand_r(int *p, const int N, const int r) {
    int *q = new int[N];
    rand_permutation(q, N);
    std::copy(q, q + r, p);
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
        std::cout << "-o <filename>: set the sa file name" << std::endl;
        std::cout << "-s <int>: set random seed" << std::endl;
        std::cout << "-r <int>: rank" << std::endl;
        std::cout << "-t <str>: the transform: one of {fwht, dft, dct}" << std::endl;
        std::cout << "-d <str>: number of elements to sample. must be <= N" << std::endl;
        std::cout << "-B <str>: number of columns of A (default B=1)" << std::endl;
        return 0;
    }

    // Open sa File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

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
    int B = find_int_arg(argc, argv, "-B", 1);
    int N = 1;
    while (N <= n) N *= 2;
    assert(d <= n);
    gen.seed(seed);
    // if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwt"))
    //n = 1 << n;
    //d = 1 << d;
    // Complex *arr = new Complex[n];
    // rand_complex_array(arr, n, part_seed);

    double* a = new double[N];
    double* sa_re = new double[d], *sa_im = new double[d];
    int* perm = new int[N]; rand_permutation(perm, N);
    int* r = new int[d]; rand_r(r, N, d);
    int* f = new int[N]; rand_sign_array(f, N);

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
    Transform transform;
    if (!strcmp(ttype, "fwht") || !strcmp(ttype, "fwhts")) {
        transform = Transform::walsh;
    } else if (!strcmp(ttype, "dft") || !strcmp(ttype, "dfts")) {
        transform = Transform::fourier;
    } else if (!strcmp(ttype, "dct") || !strcmp(ttype, "dcts")) {
        transform = Transform::cosine;
    } else {
        std::cout << "Not a supported transform type!\n";
        return -1;
    }
#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
    {
        for (int b = 0; b < B; ++b) {

#ifdef _OPENMP
#pragma omp master
#endif
            {
                rand_double_array(a, n, N);
            }
            if (!strcmp(ttype, "fwht") || !strcmp(ttype, "dft") || !strcmp(ttype, "dct")) {
                srft(N, d, n_ranks, f, perm, a, sa_re, sa_im, r, transform);
            } else {
                srft_nlogd(N, d, n_ranks, f, perm, a, sa_re, sa_im, r, transform);
            }

#ifdef _OPENMP
#pragma omp master
#endif
            {
                if (fsave.good()) {
                    fsave << b << "-th column (real part): ";
                    for (int i = 0; i < d; ++i) {
                        fsave << sa_re[i] << " ";
                    }
                    fsave << std::endl;
                    if (transform == Transform::fourier) {
                        fsave << b << "-th column (imaginary part): ";
                        for (int i = 0; i < d; ++i) {
                            fsave << sa_im[i] << " ";
                        }
                        fsave << std::endl;
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    delete[] a;
    delete[] sa_re;
    delete[] sa_im;
    delete[] perm;
    delete[] r;
    delete[] f;
    // Finalize
    std::cout << std::fixed << "Simulation Time = " << seconds << " seconds for arr of size " << n << " * " << B << " using transform " << ttype << " with seed " << seed << " and d " << d << " and #ranks " << n_ranks << ".\n";
    // fsave.close();
}
