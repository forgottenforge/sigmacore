#pragma once

#include <vector>
#include <random>

namespace sigma_c {

class StatsEngine {
public:
    static std::pair<double, double> bootstrap_ci(
        const std::vector<double>& data,
        int n_reps = 1000,
        double ci_level = 0.95,
        int seed = 42
    );
};

} // namespace sigma_c
