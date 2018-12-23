#include <cmath>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>
#include <random>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "cl2.hpp"

#include "saxpy.hpp"

cl::Program saxpy_program;
cl::Context context = cl::Context::getDefault();
cl::CommandQueue queue = cl::CommandQueue::getDefault();

class SAXPY_OCL_HOST_PTR : public SAXPYBase {
    typedef cl::KernelFunctor<cl::Buffer, cl::Buffer, float> kernel_t;
    kernel_t kernel;
public:
    SAXPY_OCL_HOST_PTR(float a, int N) : SAXPYBase(a, N), kernel(saxpy_program, "saxpy") {}
protected:
    void _compute_in_gpu() {
        cl::Buffer src(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS, N * sizeof(float), x);
        cl::Buffer dst(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_READ_ONLY, N * sizeof(float), y_gpu);
        kernel(cl::EnqueueArgs(cl::NDRange(N)), src, dst, a).wait();
    }
};

class SAXPY_OCL_SVM : public SAXPYBase {
    typedef cl::KernelFunctor<float*, float*, float> kernel_t;
    kernel_t kernel;
public:
    SAXPY_OCL_SVM(float a, int N) : SAXPYBase(a, N), kernel(saxpy_program, "saxpy") {}
    ~SAXPY_OCL_SVM() {
        _free();
    }
protected:
    void _compute_in_gpu() {
        queue.enqueueUnmapSVM(x);
        queue.enqueueUnmapSVM(y_gpu);
        kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), x, y_gpu, a).wait();
        queue.enqueueMapSVM(x, CL_TRUE, CL_MAP_WRITE, N * sizeof(float));
        queue.enqueueMapSVM(y_gpu, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, N * sizeof(float));
    }
    void _alloc() {
        x = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE, N * sizeof(float), 0);
        y = new float[N];
        y_gpu = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE, N * sizeof(float), 0);
        queue.enqueueMapSVM(x, CL_TRUE, CL_MAP_WRITE, N * sizeof(float));
        queue.enqueueMapSVM(y_gpu, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, N * sizeof(float));
    }
    void _free() {
        if (x) clSVMFree(context(), x);
        if (y) delete[] y;
        if (y_gpu) clSVMFree(context(), y_gpu);
        x = y = y_gpu = nullptr;
    }
};

class SAXPY_OCL_SVM_FINE_GRAINED : public SAXPYBase {
    typedef cl::KernelFunctor<float*, float*, float> kernel_t;
    kernel_t kernel;
public:
    SAXPY_OCL_SVM_FINE_GRAINED(float a, int N) : SAXPYBase(a, N), kernel(saxpy_program, "saxpy") {}
    ~SAXPY_OCL_SVM_FINE_GRAINED() {
        _free();
    }
protected:
    void _compute_in_gpu() {
        kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), x, y_gpu, a).wait();
    }
    void _alloc() {
        x = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, N * sizeof(float), 0);
        y = new float[N];
        y_gpu = (float*)clSVMAlloc(context(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, N * sizeof(float), 0);
    }
    void _free() {
        if (x) clSVMFree(context(), x);
        if (y) delete[] y;
        if (y_gpu) clSVMFree(context(), y_gpu);
        x = y = y_gpu = nullptr;
    }
};

class SAXPY_OCL_SVM_SYSTEM : public SAXPYBase {
    typedef cl::KernelFunctor<float*, float*, float> kernel_t;
    kernel_t kernel;
public:
    SAXPY_OCL_SVM_SYSTEM(float a, int N) : SAXPYBase(a, N), kernel(saxpy_program, "saxpy") {}
protected:
    void _compute_in_gpu() {
        kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), x, y_gpu, a).wait();
    }
};

int main() {
    std::string code{R"CLC(
        __kernel void saxpy(__global float *src, __global float *dst, float factor) {
            long i = get_global_id(0);
            dst[i] += src[i] * factor;
        }
    )CLC"};

    auto device = cl::Device::getDefault();
    saxpy_program = cl::Program(code);
    saxpy_program.build("-cl-std=CL2.0");

    const auto a = 100.0f;
    const auto N = 1024 * 1024 * 256;
    std::vector<std::tuple<std::string, std::function<Result()>>> functions;
    auto cap = device.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
    if (1)
        functions.push_back(std::make_tuple("HOST_PTR          ", std::bind(&run<SAXPY_OCL_HOST_PTR>, a, N)));
    if (cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
        functions.push_back(std::make_tuple("SVM Coarse-grained", std::bind(&run<SAXPY_OCL_SVM>, a, N)));
    if (cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
        functions.push_back(std::make_tuple("SVM Fine-grained  ", std::bind(&run<SAXPY_OCL_SVM_FINE_GRAINED>, a, N)));
    if (cap & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
        functions.push_back(std::make_tuple("SVM System        ", std::bind(&run<SAXPY_OCL_SVM_SYSTEM>, a, N)));
    std::vector<std::vector<Result>> results;
    results.resize(functions.size());
    for (auto tries = 0; tries < 3; ++tries) {
        for (auto i = 0u; i < functions.size(); ++i) {
            results[i].push_back(std::get<1>(functions[i])());
        }
    }
    for (auto i = 0u; i < functions.size(); ++i) {
        Result::dump(std::get<0>(functions[i]), results[i]);
    }
}
