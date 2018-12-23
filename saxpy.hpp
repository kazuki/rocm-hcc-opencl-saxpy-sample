#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <memory>

struct Result {
    double init_time;
    double cpu_time;
    double gpu_time;
    double verify_time;
    int errors;

    static std::tuple<Result, Result> summarize(const std::vector<Result>& results) {
        Result avg {};
        for (auto r : results) {
            avg.init_time += r.init_time;
            avg.cpu_time += r.cpu_time;
            avg.gpu_time += r.gpu_time;
            avg.verify_time += r.verify_time;
            avg.errors += r.errors;
        }
        avg.init_time /= results.size();
        avg.cpu_time /= results.size();
        avg.gpu_time /= results.size();
        avg.verify_time /= results.size();
        avg.errors /= results.size();

        Result var {};
        for (auto r : results) {
            var.init_time += std::pow(r.init_time - avg.init_time, 2);
            var.cpu_time += std::pow(r.cpu_time - avg.cpu_time, 2);
            var.gpu_time += std::pow(r.gpu_time - avg.gpu_time, 2);
            var.verify_time += std::pow(r.verify_time - avg.verify_time, 2);
            var.errors += std::pow(r.errors - avg.errors, 2);
        }
        var.init_time /= results.size();
        var.cpu_time /= results.size();
        var.gpu_time /= results.size();
        var.verify_time /= results.size();
        var.errors /= results.size();
        return std::make_tuple(avg, var);
    }

    static void dump(const std::string& name, const std::vector<Result>& results) {
        auto [avg, var] = Result::summarize(results);
        std::cout
            << "[" << name << "] cpu:"
            << std::fixed << std::setfill('0') << std::setprecision(4) << std::setw(6)
            << avg.cpu_time << "(" << var.cpu_time << ")  "
            << "init:" << avg.init_time << "(" << var.init_time << ")  "
            << "gpu:" << avg.gpu_time << "(" << var.gpu_time << ")  "
            << "verify:" << avg.verify_time << "(" << var.verify_time << ")";
        if (avg.errors != 0 || var.errors != 0)
            std::cout << "  errors:" << avg.errors;
        std::cout << std::endl;
    }
};

class SAXPYBase {
protected:
    float a;
    float *x;
    float *y;
    float *y_gpu;
    int N;

    virtual void _compute_in_gpu() = 0;
public:
    SAXPYBase(float a, int N)
        : a(a), x(nullptr), y(nullptr), y_gpu(nullptr), N(N) {}
    virtual ~SAXPYBase() {
        _free();
    }

    double init() {
        std::mt19937 random_gen(0);
        std::uniform_real_distribution<float> distribution(-N, N);
        auto tmp0 = new float[N];
        auto tmp1 = new float[N];
        std::generate_n(tmp0, N, [&]() { return distribution(random_gen); });
        std::generate_n(tmp1, N, [&]() { return distribution(random_gen); });
        auto start = std::chrono::high_resolution_clock::now();
        _free();
        _alloc();
        std::copy_n(tmp0, N, x);
        std::copy_n(tmp1, N, y);
        std::copy_n(tmp1, N, y_gpu);
        auto end = std::chrono::high_resolution_clock::now();
        delete[] tmp0;
        delete[] tmp1;
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }

    double compute_in_cpu() {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            y[i] += a * x[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }

    double compute_in_gpu() {
        auto start = std::chrono::high_resolution_clock::now();
        this->_compute_in_gpu();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }

    std::tuple<int, double> verify() {
        int errors = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            if (fabs(y[i] - y_gpu[i]) > fabs(y[i] * 0.0001f))
                errors++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return std::make_tuple(errors, diff.count());
    }
protected:
    virtual void _alloc() {
        x = new float[N];
        y = new float[N];
        y_gpu = new float[N];
    }
    virtual void _free() {
        if (x) delete[] x;
        if (y) delete[] y;
        if (y_gpu) delete[] y_gpu;
        x = y = y_gpu = nullptr;
    }
};

template<class T>
Result run(float a, int N) {
    auto instance = std::make_unique<T>(a, N);
    auto init_time = instance->init();
    auto cpu_time = instance->compute_in_cpu();
    auto gpu_time = instance->compute_in_gpu();
    auto [errors, verify_time] = instance->verify();
    return Result {
        .init_time = init_time,
        .cpu_time = cpu_time,
        .gpu_time = gpu_time,
        .verify_time = verify_time,
        .errors = errors
    };
}
