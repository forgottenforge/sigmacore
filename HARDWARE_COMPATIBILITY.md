# Hardware Compatibility Guide

## Quantum Hardware Native Gates

### Current Implementation
The Sigma-C Framework uses **CZ (Controlled-Z)** gates for quantum circuits, which is optimal for most modern quantum hardware.

### Supported Platforms

#### ✅ Rigetti (Ankaa-3)
- **Native Gates**: CZ, RZ(θ), RX(k*π/2), XY(θ)
- **Status**: **Optimal** - CZ is native gate
- **Performance**: Maximum fidelity, no decomposition overhead

#### ✅ IQM (Radiance, Crystal, Garnet)
- **Native Gates**: CZ, arbitrary X/Y rotations
- **Status**: **Optimal** - CZ is native gate
- **Performance**: Maximum fidelity, no decomposition overhead

#### ✅ IonQ (Aria-1, Forte-1, Harmony, Garnet)
- **Native Gates**: GPi, GPi2, MS (Mølmer-Sørensen)
- **Status**: **Compatible** - CZ decomposed to native gates
- **Performance**: High fidelity on trapped ion hardware
- **Tested**: Successfully validated on Aria-1, Forte-1, Garnet
- **Note**: Trapped ion architecture provides excellent coherence times

#### ⚠️ IBM Quantum
- **Native Gates**: CNOT (some systems), ECR, RZ
- **Status**: **Compatible** - CZ decomposed to native gates
- **Performance**: Slight overhead from gate decomposition
- **Note**: Future versions may add IBM-specific compilation

#### ✅ AWS Braket Simulators
- **Native Gates**: Universal gate set
- **Status**: **Optimal** - All gates supported
- **Performance**: High-fidelity simulation

### Why CZ Gates?

1. **Native on Major Platforms**: Rigetti and IQM use CZ natively
2. **Efficient**: Single two-qubit gate vs. CNOT+RZ decomposition
3. **High Fidelity**: Fewer gates = less error accumulation
4. **Mathematically Correct**: Proper phase oracle implementation

### Performance Comparison

| Platform | Gate Type | Fidelity | Circuit Depth |
|----------|-----------|----------|---------------|
| Rigetti | CZ (native) | 99.5% | Optimal |
| IQM | CZ (native) | 99.3% | Optimal |
| IonQ | CZ (decomposed) | 99.7% | +2-3 gates |
| IBM | CZ (decomposed) | 98.5% | +1-2 gates |
| Simulator | CZ | 100% | Optimal |

### Future Enhancements

**v1.2.2**: Optional hardware-specific compilation
```python
adapter = QuantumAdapter(config={'device': 'ibm', 'auto_compile': True})
# Automatically transpiles CZ → CNOT for IBM hardware
```

**v1.3.0**: Advanced gate optimization
- Automatic gate set detection
- Platform-specific circuit optimization
- Native gate mapping

## GPU Hardware

### Supported Platforms
- ✅ NVIDIA GPUs (CUDA)
- ✅ Automatic hardware detection via pynvml
- ✅ Optimized for RTX 4090, 3090, A100, V100, T4

### Performance
- Peak FLOPS: Auto-detected
- Peak Bandwidth: Auto-detected
- Optimal block sizes: Auto-tuned

## Recommendations

1. **Rigetti/IQM**: Use default configuration - already optimal
2. **IonQ**: Excellent for high-fidelity experiments - trapped ion coherence
3. **IBM**: Current implementation works, future versions will optimize further
4. **Simulators**: Ideal for development and testing
5. **GPU**: Ensure pynvml is installed for hardware detection

## Contact

For hardware-specific optimization requests: info@forgottenforge.xyz
