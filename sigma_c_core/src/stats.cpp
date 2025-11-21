#include "sigma_c/stats.hpp"
#include <algorithm>
#include <cmath>

namespace sigma_c {

std::pair<double, double> StatsEngine::bootstrap_ci(
    const std::vector<double>& data,
    int n_reps,
    double ci_level,
    int seed
) {
    if (data.empty()) return {0.0, 0.0};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    
    std::vector<double> means(n_reps);
    size_t n = data.size();
    
    for (int i = 0; i < n_reps; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            sum += data[dist(rng)];
        }
        means[i] = sum / n;
    }
    
    std::sort(means.begin(), means.end());
    
    double alpha = (1.0 - ci_level) / 2.0;
    size_t idx_lower = static_cast<size_t>(alpha * n_reps);
    size_t idx_upper = static_cast<size_t>((1.0 - alpha) * n_reps);
    
    idx_lower = std::min(idx_lower, static_cast<size_t>(n_reps - 1));
    idx_upper = std::min(idx_upper, static_cast<size_t>(n_reps - 1));
    
    return {means[idx_lower], means[idx_upper]};
}

} // namespace sigma_c
