# LLM Drift Demo - README

## Quick Start

1. **Run the simulation**:
   ```bash
   python drift_simulation.py
   ```

2. **View results**:
   - Open `dashboard.html` in your browser
   - Read `demo_guide.md` for detailed explanation

## What This Demo Shows

This demo proves that **Sigma-C can detect LLM model drift 2+ minutes before traditional monitoring**, and use **Active Control** to prevent system collapse.

## Files

- `drift_simulation.py` - Main simulation script
- `dashboard.html` - Interactive visualization dashboard
- `demo_guide.md` - Comprehensive tutorial and explanation
- `simulation_results.json` - Generated data (after running simulation)

## Requirements

- Python 3.8+
- Sigma-C framework (installed from parent directory)
- Modern web browser (for dashboard)

## The Key Insight

> **"Traditional monitoring tells you your system is dead. Sigma-C tells you it's dying — and gives you the tools to save it."**

Traditional MLOps tools measure **performance** (accuracy, latency). They detect problems **after** they happen.

Sigma-C measures **susceptibility** — how close your system is to a critical point. This gives you **early warning** before performance degrades.

---

Copyright (c) 2025 ForgottenForge.xyz
