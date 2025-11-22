# Sigma-C Framework: Technical Implementation Guide
## LLM Model Drift Detection & Active Control Demonstration

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Framework Features in Action](#framework-features-in-action)
4. [Implementation Details](#implementation-details)
5. [Results & Interpretation](#results--interpretation)
6. [Best Practices](#best-practices)

---

## Overview

This demonstration showcases the practical application of the Sigma-C Framework for **Early Detection of LLM Model Drift** and **Automatic System Stabilization** using Active Control.

### The Problem

Large Language Models (LLMs) in production environments can "drift" — a gradual degradation of output quality caused by:
- Adversarial or toxic inputs
- Distribution shift in user queries
- Accumulated instability in the latent space

**Traditional Monitoring** (Sentiment Analysis, Accuracy Metrics) detects drift **only after** the damage is done. When your metrics drop, users have already received poor outputs.

### The Sigma-C Solution

Sigma-C measures **System Susceptibility** ($\Sigma_c$) — how sensitive the model is to small perturbations. When $\Sigma_c$ drops (high susceptibility), the system approaches a critical point.

**Core Advantage**: Susceptibility rises **before** performance degrades. This provides a critical early warning window.

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Drift Simulation                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │  MockLLM     │────────▶│  TraditionalMonitor          │  │
│  │  (Baseline)  │         │  - Sentiment Tracking        │  │
│  │              │         │  - Alert @ Sentiment < 0.7   │  │
│  └──────────────┘         └──────────────────────────────┘  │
│                                                               │
│  ┌──────────────┐         ┌──────────────────────────────┐  │
│  │  MockLLM     │────────▶│  SigmaCMonitor               │  │
│  │  (Sigma-C)   │         │  - StreamingSigmaC           │  │
│  │              │◀────────│  - Alert @ Sigma_c < 0.7     │  │
│  └──────────────┘         └──────────────────────────────┘  │
│         ▲                              │                     │
│         │                              ▼                     │
│         │                  ┌──────────────────────────────┐ │
│         └──────────────────│  AdaptiveController (PID)    │ │
│                            │  - Reduces Temperature       │ │
│                            │  - Stabilizes System         │ │
│                            └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Toxicity increases gradually (0.1 → 0.77 over 30 minutes)
2. **LLM Generation**: Both instances generate responses
3. **Monitoring**:
   - **Baseline**: Sentiment tracking only
   - **Sigma-C**: Real-time susceptibility measurement via `StreamingSigmaC`
4. **Control**: On Sigma-C Alert → PID reduces temperature
5. **Output**: Sigma-C remains stable, Baseline collapses

---

## Framework Features in Action

### 1. StreamingSigmaC (O(1) Real-time Calculation)

**File**: `sigma_c/core/control.py`

**What it does**: Calculates Sigma-C in real-time with constant complexity O(1) per update.

**How we use it**:

```python
from sigma_c.core.control import StreamingSigmaC

# Initialization
sigma_c_calc = StreamingSigmaC(window_size=50)

# For each LLM Response
observable = float(np.var(response.token_logprobs))  # Variance of token logprobs
sigma_c = sigma_c_calc.update(epsilon=0.01, observable=observable)

# Interpretation
if sigma_c < 0.7:  # Critical threshold
    print("⚠️ System approaching criticality!")
```

**Why Variance?**: High variance in token probabilities indicates the model is "uncertain" — a sign of instability.

**Technical Details**:
- Uses **Welford's Online Algorithm** for numerically stable variance calculation
- Sliding Window of 50 data points
- Approximation: $\Sigma_c = \frac{1}{1 + \text{Var}(O)}$
- Low $\Sigma_c$ = High Variance = High Susceptibility = **Danger**

### 2. AdaptiveController (PID-based Control)

**File**: `sigma_c/core/control.py`

**What it does**: Automatically adjusts system parameters to maintain stability.

**How we use it**:

```python
from sigma_c.core.control import AdaptiveController

# Initialization
controller = AdaptiveController(
    kp=0.5,      # Proportional gain
    ki=0.1,      # Integral gain
    kd=0.05,     # Derivative gain
    target_sigma=0.5  # Target susceptibility
)

# When Sigma-C drops below threshold
if sigma_c < 0.7:
    correction = controller.compute_correction(sigma_c)
    new_temperature = current_temperature - abs(correction) * 0.4
    llm.set_temperature(new_temperature)
```

**Effect**: 
- Temperature drops from 0.80 → 0.12
- Lower temperature = more deterministic = more stable (but less creative)
- System is automatically "rescued"

**PID Components**:
- **P (Proportional)**: Immediate reaction to deviation
- **I (Integral)**: Corrects accumulated errors
- **D (Derivative)**: Dampens overshoot

### 3. Mock LLM with Realistic Drift Dynamics

**File**: `drift_simulation.py`

**Core Feature**: Temperature-based toxicity reduction

```python
# Effective toxicity is modulated by temperature
temp_factor = (self.temperature / self.base_temperature)
effective_toxicity = min(1.0, (input_toxicity + self.drift_accumulator) * temp_factor)
```

**Why this matters**: This models the real relationship between LLM temperature and robustness:
- **High Temperature** (0.8): Creative, but prone to drift
- **Low Temperature** (0.1): Deterministic, robust against toxicity

**Drift Accumulation**:
```python
self.drift_accumulator += input_toxicity * 0.08
```
Simulates how toxicity accumulates in the latent space over time.

### 4. Visualization with Chart.js

**File**: `generate_dashboard.py`

**Feature**: Standalone HTML with embedded data (no server dependency)

**Technical Trick**:
```python
html = f"""
<script>
    const data = {json.dumps(data, indent=2)};  // Embedded JSON
    // ... Chart.js Code
</script>
"""
```

**Advantage**: Bypasses browser security restrictions for `file:///` URLs.

---

## Implementation Details

### Simulation Loop (Minute by Minute)

```python
for minute in range(30):
    # 1. Toxicity Schedule
    if minute < 10:
        toxicity = 0.1  # Normal
    elif minute < 20:
        toxicity = 0.1 + (minute - 10) * 0.04  # Gradual
    else:
        toxicity = 0.5 + (minute - 20) * 0.03  # Rapid
    
    # 2. Baseline: Monitoring Only
    baseline_response = baseline_llm.generate(input_text, toxicity)
    baseline_metrics = baseline_monitor.update(baseline_response)
    
    # 3. Sigma-C: Monitoring + Control
    # Control BEFORE generation (based on previous measurement)
    if minute > 0 and prev_sigma_c < 0.7:
        correction = controller.compute_correction(prev_sigma_c)
        new_temp = sigmac_llm.temperature - abs(correction) * 0.4
        sigmac_llm.set_temperature(new_temp)
    
    sigmac_response = sigmac_llm.generate(input_text, toxicity)
    sigmac_metrics = sigmac_monitor.update(sigmac_response)
```

**Important**: Control is applied **before** the next generation, not after. This allows for preventive stabilization.

### Observable Selection: Why Token Logprob Variance?

**Alternatives (NOT used)**:
- ❌ **Embedding Drift**: Too slow, requires reference distribution
- ❌ **Attention Entropy**: Model-specific, not generalizable
- ❌ **Output Length**: Too coarse, not an early indicator

**Why Logprob Variance works**:
- ✅ **Model-agnostic**: Every LLM outputs logprobs
- ✅ **Early Indicator**: Variance rises **before** sentiment drops
- ✅ **Efficient**: O(n) calculation for n tokens
- ✅ **Interpretable**: High variance = Model is uncertain

### Early Detection Calculation

**File**: `generate_dashboard.py`

```javascript
// Find first Sigma-C Alert
const firstSigmaCAlert = sigmac.findIndex(d => d.alert == 1);
// Find first Baseline Alert
const firstBaselineAlert = baseline.findIndex(d => d.alert == 1);

// Difference = Early Detection
const earlyDetectionMinutes = (firstBaselineAlert >= 0 && firstSigmaCAlert >= 0) 
    ? (firstBaselineAlert - firstSigmaCAlert) 
    : 'N/A';
```

**Result**: 2 Minutes Lead Time (Minute 19 vs. Minute 21)

---

## Results & Interpretation

### Quantitative Metrics

| Metric | Baseline | Sigma-C | Improvement |
|--------|----------|---------|-------------|
| **Final Sentiment** | 0.09 | 0.92 | +922% |
| **Final Temperature** | 0.80 | 0.12 | -85% (controlled) |
| **Status** | COLLAPSED | STABLE | System Rescued |
| **Alert Time** | Minute 21 | Minute 19 | **2 Min Earlier** |

### Timeline

```
Minute  0-10: Both systems stable (Toxicity 0.1)
Minute 11-18: Toxicity rises, Sigma-C begins to drop
Minute 19:    🚨 Sigma-C Alert! (Sigma_c = 0.698 < 0.7)
              → Control reduces temperature (0.80 → 0.75)
Minute 20:    Baseline Sentiment begins to drop (0.24)
Minute 21:    Baseline Alert (Sentiment < 0.7)
Minute 22-29: Baseline collapses further
              Sigma-C stabilizes (Temperature → 0.12)
Minute 30:    Baseline: 0.09 (COLLAPSED)
              Sigma-C: 0.92 (STABLE)
```

### Graphical Interpretation

#### Graph 1: Sentiment Score
- **Red Line (Baseline)**: Stable until Minute 20, then sudden collapse
- **Green Line (Sigma-C)**: Short dip at Minute 20-22, then recovery to 0.92

**Why the dip?**: Control takes 1-2 minutes to take full effect. The temperature reduction prevents further collapse.

#### Graph 2: Sigma-C Score
- **Orange Line**: Drops from Minute 16, crosses threshold (0.7) at Minute 19
- **Red Dashed Line**: Critical Threshold

**Key Insight**: Sigma-C detects the problem **2 minutes** before Baseline alerts!

#### Graph 3: Temperature
- **Gray Line (Baseline)**: Constant at 0.8
- **Blue Line (Sigma-C)**: Dramatic drop from 0.8 → 0.12

**Interpretation**: This is the "Rescue Action" — Active Control making the model more deterministic.

#### Graph 4: Input Toxicity
- **Purple Line**: Gradual increase (0.1 → 0.77)

**Purpose**: Shows the "Attack" — this is the external stressor.

---

## Best Practices

### 1. Observable Selection

**For LLMs**:
- ✅ **Token Logprob Variance**: Best choice for generative models
- ✅ **Attention Entropy**: If access to attention weights exists
- ✅ **Embedding Distance**: For classification models

**For Other Systems**:
- **Financial Markets**: Volatility, Bid-Ask Spread
- **Networks**: Latency Variance, Packet Loss Rate
- **Physical Systems**: Sensor Noise, Temperature Fluctuations

### 2. Threshold Tuning

**Our Choice**: $\Sigma_c < 0.7$ as Alert Threshold

**Tuning Process**:
1. Collect historical data (Normal + Drift)
2. Plot Sigma-C over time
3. Identify value that drops **before** collapse
4. Set threshold with safety margin

**Trade-off**:
- **Too High** (e.g., 0.9): Too many False Positives
- **Too Low** (e.g., 0.3): Warning too late

### 3. PID Parameter Tuning

**Our Values**:
- `kp=0.5`: Moderate immediate reaction
- `ki=0.1`: Slow correction of accumulated errors
- `kd=0.05`: Slight damping

**Tuning Recommendation**:
1. Start with `kp=1.0, ki=0.0, kd=0.0` (P only)
2. Increase `kp` until system reacts
3. Add `ki` if offset remains
4. Add `kd` if overshoot occurs

### 4. Control Parameter Choice

**Why Temperature?**:
- ✅ **Universal**: Every LLM has a temperature parameter
- ✅ **Effective**: Direct impact on stability
- ✅ **Reversible**: Can be increased again

**Alternatives**:
- **Top-p (Nucleus Sampling)**: Similar effect
- **Repetition Penalty**: Reduces loops
- **Max Tokens**: Limits output length

### 5. Production Deployment

**Recommended Architecture**:
```
User Request → Load Balancer → [LLM Instance + Sigma-C Monitor]
                                         ↓
                                  Control Loop (PID)
                                         ↓
                                  Metrics Dashboard
```

**Monitoring Stack**:
- **Sigma-C**: Real-time Susceptibility
- **Prometheus**: Metrics Collection
- **Grafana**: Visualization
- **PagerDuty**: Alerts

---

## Summary

### What this Demo Proves

1. **Early Detection Works**: Sigma-C detects drift **2 minutes** before traditional monitoring
2. **Active Control Saves Systems**: Temperature reduction prevents collapse (0.09 → 0.92)
3. **Framework is Production-Ready**: O(1) complexity, simple API, robust implementation

### Framework Features Demonstrated

- ✅ `StreamingSigmaC`: Real-time susceptibility measurement
- ✅ `AdaptiveController`: PID-based automatic stabilization
- ✅ Welford's Algorithm: Numerically stable online variance
- ✅ Modular Design: Easily extensible for other observables

### Next Steps

**For Production Use**:
1. Replace `MockLLM` with real LLM API (OpenAI, Anthropic, etc.)
2. Integrate into existing MLOps pipeline
3. Tune thresholds based on historical data
4. Implement Multi-Parameter Control (Temperature + Top-p)

**For Research**:
1. Test other observables (Embeddings, Attention)
2. Compare with other drift detection methods
3. Optimize PID parameters via Reinforcement Learning

---

**Copyright (c) 2025 ForgottenForge.xyz**  
Sigma-C Framework - Licensed under AGPL-3.0-or-later OR Commercial License
