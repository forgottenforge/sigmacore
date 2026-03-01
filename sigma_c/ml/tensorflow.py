"""
Sigma-C TensorFlow/Keras Integration
======================================
Copyright (c) 2025 ForgottenForge.xyz

Criticality-aware neural network training for TensorFlow/Keras.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Dict, Any, Optional, List
import numpy as np

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    tf = None


class CriticalModel:
    """
    Wrapper that adds criticality tracking to any Keras model.

    Usage:
        from sigma_c.ml.tensorflow import CriticalModel

        base_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model = CriticalModel(base_model)
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=10)

        report = model.get_criticality_report()
    """

    def __init__(self, model):
        if not _HAS_TF:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.model = model
        self.sigma_c_history = []
        self.enable_criticality_tracking = True

    def compile(self, **kwargs):
        """Compile the underlying Keras model."""
        self.model.compile(**kwargs)

    def fit(self, *args, **kwargs):
        """
        Train with automatic criticality tracking via callback.
        """
        callbacks = kwargs.get('callbacks', []) or []
        if self.enable_criticality_tracking:
            callbacks.append(SigmaCCallback(self))
        kwargs['callbacks'] = callbacks
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Forward prediction."""
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the model."""
        return self.model.evaluate(*args, **kwargs)

    def _compute_activation_criticality(self) -> float:
        """
        Compute criticality from layer weight statistics.

        Near criticality: weight distributions show high variance relative
        to their mean, indicating the network is near a phase transition
        in its loss landscape.
        """
        all_weights = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            for w in weights:
                all_weights.append(w.flatten())

        if not all_weights:
            return 0.0

        concatenated = np.concatenate(all_weights)
        var = np.var(concatenated)
        mean_abs = np.mean(np.abs(concatenated))

        if mean_abs > 0:
            sigma_c = var / (mean_abs + 1e-9)
        else:
            sigma_c = 0.0

        return float(np.clip(sigma_c, 0, 1))

    def get_criticality_report(self) -> Dict[str, Any]:
        """
        Get criticality statistics over training.

        Returns:
            Dictionary with sigma_c statistics across epochs.
        """
        if not self.sigma_c_history:
            return {'mean_sigma_c': 0.0, 'std_sigma_c': 0.0, 'samples': 0}

        history = np.array(self.sigma_c_history)
        return {
            'mean_sigma_c': float(np.mean(history)),
            'std_sigma_c': float(np.std(history)),
            'min_sigma_c': float(np.min(history)),
            'max_sigma_c': float(np.max(history)),
            'samples': len(history),
        }


class SigmaCCallback:
    """
    Keras callback for tracking criticality during training.

    Usage:
        callback = SigmaCCallback()
        model.fit(x, y, callbacks=[callback])
        print(callback.history)

    Can also be used standalone with any Keras model:
        callback = SigmaCCallback()
        model.fit(x, y, callbacks=[callback])
        print(callback.history)
    """

    def __init__(self, critical_model: Optional[CriticalModel] = None):
        self._critical_model = critical_model
        self._model = None
        self.history: List[Dict[str, float]] = []

    def set_model(self, model):
        """Called by Keras at the start of training."""
        self._model = model

    def set_params(self, params):
        """Called by Keras at the start of training."""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Compute criticality at end of each epoch."""
        model = self._model
        if model is None and self._critical_model is not None:
            model = self._critical_model.model

        if model is None:
            return

        all_weights = []
        for layer in model.layers:
            weights = layer.get_weights()
            for w in weights:
                all_weights.append(w.flatten())

        if not all_weights:
            return

        concatenated = np.concatenate(all_weights)
        var = float(np.var(concatenated))
        mean_abs = float(np.mean(np.abs(concatenated)))
        sigma_c = float(np.clip(var / (mean_abs + 1e-9), 0, 1)) if mean_abs > 0 else 0.0

        entry = {
            'epoch': epoch,
            'sigma_c': sigma_c,
            'weight_variance': var,
            'weight_mean_abs': mean_abs,
            'loss': float(logs.get('loss', 0.0)) if logs else 0.0,
        }
        self.history.append(entry)

        if self._critical_model is not None:
            self._critical_model.sigma_c_history.append(sigma_c)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class CriticalRegularizer:
    """
    Keras regularizer that penalizes deviation from target criticality.

    Usage:
        from sigma_c.ml.tensorflow import CriticalRegularizer

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64,
                kernel_regularizer=CriticalRegularizer(target_sigma_c=0.5)),
            tf.keras.layers.Dense(1)
        ])
    """

    def __init__(self, target_sigma_c: float = 0.5, strength: float = 0.01):
        if not _HAS_TF:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.target_sigma_c = target_sigma_c
        self.strength = strength

    def __call__(self, weights):
        """Compute regularization loss."""
        var = tf.math.reduce_variance(weights)
        mean_abs = tf.math.reduce_mean(tf.math.abs(weights)) + 1e-9
        sigma_c = tf.clip_by_value(var / mean_abs, 0.0, 1.0)
        return self.strength * tf.square(sigma_c - self.target_sigma_c)

    def get_config(self):
        return {
            'target_sigma_c': self.target_sigma_c,
            'strength': self.strength,
        }
