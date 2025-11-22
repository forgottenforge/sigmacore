# Sigma-C Framework: LLM Model Drift Detection & Active Control

## 🎯 Overview

This demo showcases **Sigma-C's killer feature**: detecting LLM model drift **before** it impacts performance, and using **Active Control** to prevent system collapse.

### The Problem: Toxic Drift

Large Language Models (LLMs) in production can experience "drift" — a gradual degradation in output quality caused by:
- Adversarial or toxic inputs
- Distribution shift in user queries
- Accumulated instability in the latent space

**Traditional monitoring** (sentiment analysis, accuracy metrics) only detects drift **after** the damage is done. By the time your metrics drop, users have already experienced poor outputs.

### The Sigma-C Solution

Sigma-C measures **system susceptibility** ($\Sigma_c$) — how sensitive the model is to small perturbations. When $\Sigma_c$ drops (indicating high susceptibility), the system is approaching a critical point.

**Key Insight**: Susceptibility rises **before** performance degrades. This gives you a critical early warning window.

---

## 🚀 Quick Start

### 1. Run the Simulation

```bash
cd examples_v2/llm_drift_demo
python drift_simulation.py
```

This runs a 30-minute simulation (compressed into seconds) of two LLM instances:
- **Baseline**: Traditional monitoring (sentiment only)
- **Sigma-C Protected**: Real-time susceptibility monitoring + PID control

### 2. View the Results

Open `dashboard.html` in your browser to see interactive visualizations.

---

## 📊 What to Expect

### Timeline

| Time | Event |
|------|-------|
| **Minutes 0-10** | Normal operation. Both systems stable. |
| **Minute 12** | ⚡ **Sigma-C Alert**: Susceptibility crosses threshold. Active Control reduces temperature. |
| **Minutes 12-20** | Baseline appears stable (sentiment still high), but Sigma-C knows danger is near. |
| **Minute 24** | 💥 **Baseline Collapse**: Sentiment drops below 0.5. System is producing toxic outputs. |
| **Minute 30** | **Sigma-C**: Still stable (sentiment > 0.7). Temperature reduced, but system functional. |

### Key Metrics

- **Sigma-C Score**: Lower = more critical. Threshold = 0.4.
- **Sentiment**: Proxy for output quality (0.0 = toxic, 1.0 = positive).
- **Temperature**: LLM parameter. Lower = more deterministic/safe, higher = more creative/risky.

---

## 🔬 How It Works

### 1. Susceptibility Measurement

The `StreamingSigmaC` class monitors the **variance of token log-probabilities**:

```python
from sigma_c.core.control import StreamingSigmaC

monitor = StreamingSigmaC(window_size=50)

# For each LLM response
observable = np.var(response.token_logprobs)
sigma_c = monitor.update(epsilon=0.01, observable=observable)

if sigma_c < 0.4:
    print("⚠️ System approaching criticality!")
```

**Why variance?** High variance in token probabilities indicates the model is "uncertain" — a sign of instability.

### 2. Active Control (PID)

The `AdaptiveController` adjusts the LLM's temperature to maintain stability:

```python
from sigma_c.core.control import AdaptiveController

controller = AdaptiveController(
    kp=0.5, ki=0.1, kd=0.05,
    target_sigma=0.5
)

# When Sigma-C alerts
correction = controller.compute_correction(current_sigma_c)
new_temperature = current_temperature - correction * 0.1
llm.set_temperature(new_temperature)
```

**Result**: The system automatically becomes more conservative (lower temperature) when instability is detected.

### 3. The Mock LLM

Since we can't run a real 70B parameter model in this demo, we simulate the key dynamics:

- **Input Toxicity** → **Latent Drift** → **Output Variance** → **Sentiment Degradation**
- The variance (susceptibility) rises **before** sentiment drops — this is the critical early warning.

---

## 📈 Interpreting the Graphs

### Graph 1: Sentiment Score
- **Red Line (Baseline)**: Crashes at Minute 24.
- **Green Line (Sigma-C)**: Stays stable thanks to active control.

### Graph 2: Sigma-C Score
- **Orange Line**: Drops below threshold (red dashed line) at Minute 12.
- This is **12 minutes before** the baseline collapses!

### Graph 3: Temperature
- **Blue Line (Sigma-C)**: Automatically reduces temperature when alert fires.
- **Gray Line (Baseline)**: No control — temperature stays constant.

### Graph 4: Input Toxicity
- Shows the "attack vector" — gradually increasing toxicity.

---

## 💡 Key Takeaways

### 1. Early Detection
Sigma-C detected the drift **12 minutes** before traditional monitoring. In production, this could be hours or days of advance warning.

### 2. Active Control
The PID controller automatically adjusted the system parameter (temperature) to prevent collapse. No human intervention required.

### 3. The Trade-off
Lower temperature = less creative, but more stable. Sigma-C helps you navigate this trade-off dynamically.

---

## 🛠️ Extending This Demo

### Use Real LLMs
Replace `MockLLM` with a real LLM API (OpenAI, Anthropic, etc.):

```python
import openai

class RealLLM:
    def generate(self, input_text, temperature):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
            logprobs=True
        )
        return response
```

### Monitor Other Observables
Instead of token log-probabilities, you could monitor:
- **Embedding drift**: Distance between input embeddings and training distribution
- **Attention entropy**: Variance in attention weights
- **Gradient norms**: For models you're fine-tuning

### Multi-Parameter Control
Control multiple parameters simultaneously:
- Temperature
- Top-p (nucleus sampling)
- Repetition penalty

---

## 🎓 The Science

### Fluctuation-Dissipation Theorem
Sigma-C is grounded in statistical physics. The core idea:

$$\chi = \frac{\partial \langle O \rangle}{\partial \epsilon}$$

Where:
- $\chi$ = susceptibility
- $O$ = observable (e.g., output variance)
- $\epsilon$ = perturbation (e.g., input noise)

Near a critical point, $\chi \to \infty$. Sigma-C approximates this as:

$$\Sigma_c = \frac{1}{1 + \text{Var}(O)}$$

Low $\Sigma_c$ = high variance = high susceptibility = **danger**.

### Why This Matters for LLMs
LLMs are **complex dynamical systems**. Like phase transitions in physics, they can exhibit sudden "collapses" (e.g., mode collapse, hallucination spirals). Sigma-C detects the **precursors** to these collapses.

---

## 🚨 Failed Approaches (What NOT to Do)

### ❌ Monitoring Only Accuracy/Sentiment
**Problem**: Lagging indicator. By the time accuracy drops, the model has already failed.

### ❌ Static Thresholds on Output Variance
**Problem**: Variance alone doesn't tell you if the system is *approaching* criticality or just noisy.

### ❌ Manual Temperature Tuning
**Problem**: Humans can't react fast enough. Drift can happen in minutes.

### ✅ The Right Way: Sigma-C + Active Control
**Solution**: Real-time susceptibility monitoring + automated PID control.

---

## 📚 Further Reading

- [Sigma-C Framework Documentation](../../sigma_c_framework/README.md)
- [Statistical Physics of LLMs](https://arxiv.org/abs/2304.09121) (Example paper)
- [PID Control Theory](https://en.wikipedia.org/wiki/PID_controller)

---

## 🤝 Contributing

Found a bug? Have an idea for a better observable? Open an issue or PR!

---

## 📄 License

Copyright (c) 2025 ForgottenForge.xyz  
Licensed under AGPL-3.0-or-later OR Commercial License.

---

**"Traditional monitoring tells you your system is dead. Sigma-C tells you it's dying — and gives you the tools to save it."**
