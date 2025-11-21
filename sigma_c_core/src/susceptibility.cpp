#include "sigma_c/susceptibility.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace sigma_c {

std::vector<double> SusceptibilityEngine::gaussian_smooth(
    const std::vector<double>& data,
    double sigma
) {
    int n = data.size();
    std::vector<double> smoothed(n);
    int radius = static_cast<int>(std::ceil(3 * sigma));
    
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double weight_sum = 0.0;
        
        for (int j = std::max(0, i - radius); j <= std::min(n - 1, i + radius); ++j) {
            double dist = static_cast<double>(i - j);
            double weight = std::exp(-(dist * dist) / (2 * sigma * sigma));
            sum += data[j] * weight;
            weight_sum += weight;
        }
        smoothed[i] = sum / weight_sum;
    }
    return smoothed;
}

SusceptibilityResult SusceptibilityEngine::compute(
    const std::vector<double>& epsilon,
    const std::vector<double>& observable,
    double kernel_sigma,
    double edge_damp
) {
    size_t n = epsilon.size();
    if (n < 3) {
        // Fallback for too few points
        return {std::vector<double>(n, 0.0), 0.0, 0.0, observable, 1.0};
    }

    // 1. Gaussian Smoothing
    auto smoothed = gaussian_smooth(observable, kernel_sigma);

    // 2. Numerical Gradient (Central Difference)
    std::vector<double> chi(n);
    
    // Interior points
    for (size_t i = 1; i < n - 1; ++i) {
        double dx = epsilon[i+1] - epsilon[i-1];
        double dy = smoothed[i+1] - smoothed[i-1];
        chi[i] = std::abs(dy / dx);
    }
    
    // Edges (Forward/Backward)
    chi[0] = std::abs((smoothed[1] - smoothed[0]) / (epsilon[1] - epsilon[0])) * edge_damp;
    chi[n-1] = std::abs((smoothed[n-1] - smoothed[n-2]) / (epsilon[n-1] - epsilon[n-2])) * edge_damp;

    // 3. Baseline (10th percentile of interior)
    std::vector<double> interior;
    if (n > 4) {
        for (size_t i = 1; i < n - 1; ++i) interior.push_back(chi[i]);
    } else {
        interior = chi;
    }
    
    std::sort(interior.begin(), interior.end());
    double baseline = 1e-5;
    if (!interior.empty()) {
        size_t idx = static_cast<size_t>(0.1 * interior.size());
        baseline = std::max(interior[idx], 1e-5);
    }

    // 4. Peak Detection
    auto max_it = std::max_element(chi.begin(), chi.end());
    size_t max_idx = std::distance(chi.begin(), max_it);
    
    double sigma_c = epsilon[max_idx];
    double kappa = *max_it / baseline;
    kappa = std::min(kappa, 200.0); // Clip

    return {chi, sigma_c, kappa, smoothed, baseline};
}

} // namespace sigma_c
