#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>

namespace sigma_c {

struct SusceptibilityResult {
    std::vector<double> chi;
    double sigma_c;
    double kappa;
    std::vector<double> smoothed;
    double baseline;
};

class SusceptibilityEngine {
public:
    static SusceptibilityResult compute(
        const std::vector<double>& epsilon,
        const std::vector<double>& observable,
        double kernel_sigma = 0.6,
        double edge_damp = 0.5
    );

private:
    static std::vector<double> gaussian_smooth(
        const std::vector<double>& data,
        double sigma
    );
};

} // namespace sigma_c
