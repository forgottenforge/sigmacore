optimizer = BalancedQuantumOptimizer(adapter, 0.5, 0.5)

print("Testing with minimal param space:")
result = optimizer.optimize_circuit(
    grover_factory,
    param_space={
        'epsilon': [0.0, 0.05],
        'idle_frac': [0.0]
    }
)

print(f"\nOptimal params: {result.optimal_params}")
print(f"Performance before: {result.performance_before:.4f}")
print(f"Performance after: {result.performance_after:.4f}")
print(f"Sigma_c before: {result.sigma_c_before:.4f}")
print(f"Sigma_c after: {result.sigma_c_after:.4f}")
print(f"Score: {result.score:.4f}")

print(f"\nHistory ({len(result.history)} entries):")
for i, h in enumerate(result.history):
    print(f"  {i}: params={h['params']}, perf={h['performance']:.4f}, stab={h['stability']:.4f}, score={h['score']:.4f}")
