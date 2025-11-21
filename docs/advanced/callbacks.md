# Callbacks System

## Overview
Callbacks allow you to hook into the optimization lifecycle. You can use them to:
- Log progress to a file or dashboard.
- Stop optimization early if a goal is reached.
- Save checkpoints to resume later.
- Modify parameters dynamically.

## Built-in Callbacks

### LoggingCallback
Logs the current step, score, and parameters to the console or a file.

```python
from sigma_c.core.callbacks import LoggingCallback
cb = LoggingCallback(interval=10, log_file='opt.log')
```

### EarlyStopping
Stops the optimization if the score stops improving.

```python
from sigma_c.core.callbacks import EarlyStopping
# Stop if score doesn't improve by 0.001 for 20 steps
cb = EarlyStopping(monitor='score', min_delta=0.001, patience=20)
```

### CheckpointCallback
Saves the optimizer state periodically.

```python
from sigma_c.core.callbacks import CheckpointCallback
cb = CheckpointCallback(filepath='checkpoints/model_v1', interval=50)
```

## Custom Callbacks
You can create your own callbacks by inheriting from `OptimizationCallback`.

```python
from sigma_c.core.callbacks import OptimizationCallback

class SlackNotifier(OptimizationCallback):
    def on_optimization_end(self, optimizer, result):
        send_slack_message(f"Optimization finished! Score: {result.score}")
```

### Lifecycle Methods
- `on_optimization_start(optimizer, param_space)`
- `on_step_start(optimizer, step)`
- `on_step_end(optimizer, step, logs)`
- `on_optimization_end(optimizer, result)`
