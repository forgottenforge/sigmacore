# Sigma-C Julia Package
# ======================
# Copyright (c) 2025 ForgottenForge.xyz

"""
    SigmaC

Julia package for criticality analysis.

# Examples
```julia
using SigmaC

# Compute criticality
epsilon = range(0, 0.5, length=100)
observable = sin.(epsilon * 10)
result = compute_sigma_c(epsilon, observable)

println("σ_c = ", result.sigma_c)
```
"""
module SigmaC

export compute_sigma_c, StreamingSigmaC, ObservableDiscovery

using Statistics
using FFTW

"""
    SigmaCResult

Result structure for criticality analysis.
"""
struct SigmaCResult
    sigma_c::Float64
    kappa::Float64
    chi_max::Float64
    peak_location::Float64
end

"""
    compute_sigma_c(epsilon::AbstractVector, observable::AbstractVector)

Compute critical susceptibility from data.

# Arguments
- `epsilon`: Control parameter values
- `observable`: Observable values

# Returns
- `SigmaCResult` with sigma_c, kappa, and other metrics
"""
function compute_sigma_c(epsilon::AbstractVector, observable::AbstractVector)
    # Compute susceptibility (derivative)
    chi = diff(observable) ./ diff(epsilon)
    
    # Find peak
    peak_idx = argmax(abs.(chi))
    sigma_c = epsilon[peak_idx]
    chi_max = abs(chi[peak_idx])
    
    # Compute peak sharpness (kappa)
    # Using second derivative
    if length(chi) > 2
        d2_chi = diff(chi)
        kappa = abs(d2_chi[peak_idx]) / chi_max
    else
        kappa = 1.0
    end
    
    return SigmaCResult(sigma_c, kappa, chi_max, sigma_c)
end

"""
    StreamingSigmaC

Streaming criticality calculator with O(1) updates.
"""
mutable struct StreamingSigmaC
    window_size::Int
    epsilon_buffer::Vector{Float64}
    observable_buffer::Vector{Float64}
    current_sigma_c::Float64
    
    function StreamingSigmaC(window_size::Int=100)
        new(window_size, Float64[], Float64[], 0.0)
    end
end

"""
    update!(stream::StreamingSigmaC, epsilon::Float64, observable::Float64)

Update streaming calculator with new data point.
"""
function update!(stream::StreamingSigmaC, epsilon::Float64, observable::Float64)
    push!(stream.epsilon_buffer, epsilon)
    push!(stream.observable_buffer, observable)
    
    # Keep only recent window
    if length(stream.epsilon_buffer) > stream.window_size
        popfirst!(stream.epsilon_buffer)
        popfirst!(stream.observable_buffer)
    end
    
    # Recompute sigma_c
    if length(stream.epsilon_buffer) >= 10
        result = compute_sigma_c(stream.epsilon_buffer, stream.observable_buffer)
        stream.current_sigma_c = result.sigma_c
    end
    
    return stream.current_sigma_c
end

"""
    ObservableDiscovery

Automatic observable discovery using gradient/entropy methods.
"""
struct ObservableDiscovery
    method::Symbol  # :gradient, :entropy, or :pca
end

"""
    find_optimal(discovery::ObservableDiscovery, data::Matrix)

Find optimal observable from candidate features.

# Arguments
- `data`: Matrix where each column is a candidate observable

# Returns
- Index of best observable
"""
function find_optimal(discovery::ObservableDiscovery, data::Matrix)
    n_features = size(data, 2)
    scores = zeros(n_features)
    
    for i in 1:n_features
        if discovery.method == :gradient
            # Maximize gradient norm
            scores[i] = sum(abs.(diff(data[:, i])))
        elseif discovery.method == :entropy
            # Minimize entropy
            hist = fit(Histogram, data[:, i], nbins=20)
            p = hist.weights ./ sum(hist.weights)
            p = p[p .> 0]  # Remove zeros
            scores[i] = -sum(p .* log.(p))
        end
    end
    
    return argmax(scores)
end

end # module
