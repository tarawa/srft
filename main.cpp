#include "common.h"
#include <chrono>
#include <fstream>

// =================
// Helper Functions
// =================

// make random double array
std::mt19937 gen;

int generate(unsigned seed) {
    return rand_r(&seed);
}

void rand_double_array(double *a, const int n, const int N, int seed) {
#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < n; ++i) a[i] = (generate(seed + i) / (double)RAND_MAX - 0.5) * 2;
#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = n; i < N; ++i) a[i] = 0.0;
}

void rand_sign_array(int *f, const int N) {
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < N; ++i) f[i] = (dist(gen) ? -1 : 1);
}

void rand_permutation(int *p, const int N) {
    for (int i = 0; i < N; ++i) p[i] = i;
    std::shuffle(p, p + N, gen);
}

void rand_r(int *p, const int N, const int r) {
    int *q = new int[N];
    rand_permutation(q, N);
    std::copy(q, q + r, p);
    delete[] q;
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
        std::cout << "-n <int>: array size" << std::endl;
        std::cout << "-o <filename>: set the sa file name" << std::endl;
        std::cout << "-s <int>: set random seed" << std::endl;
        std::cout << "-t <str>: the transform: one of {fwht, dft, dct}" << std::endl;
        std::cout << "-d <str>: number of elements to sample. must be <= n" << std::endl;
        std::cout << "-B <str>: number of columns of A (default B=1)" << std::endl;
        return 0;
    }

    // Open save file
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    char* ttype = find_string_option(argc, argv, "-t", nullptr);
    if (ttype == nullptr) {
        std::cout << "You must enter transform type!\n";
        exit(-1);
    }
    // Initialize

    int seed = find_int_arg(argc, argv, "-s", 0);
    int n = find_int_arg(argc, argv, "-n", 1024);
    int d = find_int_arg(argc, argv, "-d", 16);
    int B = find_int_arg(argc, argv, "-B", 1);
    int N = 1;
    int n_ranks = 1;
    while (N < n) N *= 2;
    assert(d <= n);
    gen.seed(seed);

    double* a = new double[N];
    double* sa_re = new double[d], *sa_im = new double[d];
    int* perm = new int[N]; rand_permutation(perm, N);
    int* r = new int[d]; rand_r(r, N, d);
    int* f = new int[N]; rand_sign_array(f, N);

    // Algorithm

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
    std::chrono::duration<double> rng_time;
    auto start_time = std::chrono::steady_clock::now();
    if (!strcmp(ttype, "fwht") || !strcmp(ttype, "dft") || !strcmp(ttype, "dct")) {
        init(N, d, n_ranks, f, perm, r, transform);
    } else {
        init_nlogd(N, d, n_ranks, f, perm, r, transform);
    }
#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
    {
#ifdef _OPENMP
#pragma omp master
        {
            n_ranks = omp_get_num_threads();
        }
#endif
        for (int b = 0; b < B; ++b) {
            int cur_seed;
            std::chrono::steady_clock::time_point rng_start_time, rng_end_time;
#ifdef _OPENMP
#pragma omp barrier
#pragma omp master
#endif
            {
                cur_seed = generate(seed + b);
                rng_start_time = std::chrono::steady_clock::now();
            }
#ifdef _OPENMP
#endif
            rand_double_array(a, n, N, cur_seed);
#ifdef _OPENMP
#pragma omp barrier
#pragma omp master
#endif
            {
                rng_end_time = std::chrono::steady_clock::now();
                rng_time += std::chrono::duration<double>(rng_end_time - rng_start_time);
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

    delete[] a;
    delete[] sa_re;
    delete[] sa_im;
    delete[] perm;
    delete[] r;
    delete[] f;
    // Finalize
    std::cout << std::fixed << "Simulation Time = " << (diff - rng_time).count() << " seconds for arr of size=" << n << "*" << B << " using transform " << ttype << " with seed=" << seed << " and d=" << d << " and #ranks=" << n_ranks << ". Parallel RNG Time = " << rng_time.count() << " seconds." << std::endl;
    // fsave.close();
}
