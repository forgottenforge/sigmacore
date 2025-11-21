# Configuration Guide

## Global Configuration

Sigma-C uses a hierarchical configuration system. Settings can be provided via:
1.  **Constructor Arguments** (Highest priority)
2.  **Environment Variables**
3.  **Configuration Files** (`sigma_c.yaml`)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SIGMA_C_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARN) | INFO |
| `SIGMA_C_DEVICE` | Target device (cpu, cuda, qpu) | cpu |
| `SIGMA_C_API_KEY` | API Key for cloud services (AWS/IBM) | None |

## Hardware Setup

### Quantum Computing (AWS Braket)
To use real quantum hardware or cloud simulators:

1.  Install the AWS CLI and configure credentials:
    ```bash
    aws configure
    ```
2.  Install the Braket SDK:
    ```bash
    pip install amazon-braket-sdk
    ```
3.  In your Python code:
    ```python
    from sigma_c.adapters.quantum import QuantumAdapter
    adapter = QuantumAdapter(config={'device': 'arn:aws:braket:::device/qpu/rigetti/Ankaa-2'})
    ```

### GPU Acceleration (NVIDIA)
To use GPU optimization:

1.  Ensure NVIDIA Drivers and CUDA Toolkit are installed.
2.  Install `pynvml`:
    ```bash
    pip install pynvml
    ```
3.  Sigma-C will automatically detect your GPU. You can verify this:
    ```python
    from sigma_c.adapters.gpu import GPUAdapter
    print(GPUAdapter().get_device_info())
    ```

### Machine Learning
For ML optimization, ensure you have your framework of choice installed (PyTorch, TensorFlow, or Scikit-Learn). Sigma-C is framework-agnostic but requires `scikit-learn` for some internal metrics.

```bash
pip install torch torchvision  # Example for PyTorch
```
