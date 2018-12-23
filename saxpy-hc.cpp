#include <cmath>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <utility>
#include <random>

#include <hc.hpp>
#include "saxpy.hpp"

class SAXPY_HC : public SAXPYBase {
public:
    SAXPY_HC(float a, int N) : SAXPYBase(a, N) {}
protected:
    void _compute_in_gpu() {
        auto a = this->a;
        hc::array_view<float, 1> av_x(N, x);
        hc::array_view<float, 1> av_y(N, y_gpu);
        hc::parallel_for_each(hc::extent<1>(N), [a, av_x, av_y](hc::index<1> i) [[hc]] {
            av_y[i] += a * av_x[i];
        });
    }
};

int main() {
    const auto a = 100.0f;
    const auto N = 1024 * 1024 * 256;
    auto f = std::bind(&run<SAXPY_HC>, a, N);
    std::vector<Result> results;
    for (auto tries = 0; tries < 3; ++tries) {
        results.push_back(f());
    }
    Result::dump("HCC", results);
}
