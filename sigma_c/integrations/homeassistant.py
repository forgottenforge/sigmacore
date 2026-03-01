"""
Sigma-C Home Assistant Integration
=====================================
Copyright (c) 2025 ForgottenForge.xyz

Home Assistant custom component for system criticality monitoring.
Aggregates numeric sensor readings over a sliding window and computes
a criticality score using the Sigma-C susceptibility framework.

This module provides both:
1. A standalone HomeAssistantBridge class (works without HA installed)
2. A SigmaCSensor class implementing the HA sensor protocol

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import time
from typing import Dict, Any, List
import numpy as np

from ..core.engine import Engine

try:
    from homeassistant.components.sensor import SensorEntity
    _HAS_HA = True
except ImportError:
    _HAS_HA = False
    SensorEntity = object


class HomeAssistantBridge:
    """
    Standalone bridge for Home Assistant-style sensor aggregation.

    Works without Home Assistant installed. Collects named sensor values
    over a sliding window and computes criticality from their susceptibility.

    Usage:
        from sigma_c.integrations.homeassistant import HomeAssistantBridge

        bridge = HomeAssistantBridge(window_size=50)
        bridge.add_sensor('temperature', weight=0.3)
        bridge.add_sensor('energy', weight=0.4)
        bridge.add_sensor('humidity', weight=0.3)

        # Feed data points
        for reading in data_stream:
            bridge.update('temperature', reading['temp'])
            bridge.update('energy', reading['energy'])
            bridge.update('humidity', reading['humidity'])

        state = bridge.get_state()
        print(f"sigma_c = {state['sigma_c']:.3f}")
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._engine = Engine()

    def add_sensor(self, name: str, weight: float = 1.0):
        """
        Register a sensor to monitor.

        Args:
            name: Sensor identifier (e.g. 'temperature', 'cpu_usage').
            weight: Relative weight in the combined criticality score.
        """
        self._sensors[name] = {
            'weight': weight,
            'history': [],
            'timestamps': [],
        }

    def update(self, name: str, value: float):
        """
        Push a new reading for a sensor.

        Args:
            name: Sensor identifier.
            value: Current numeric reading.
        """
        if name not in self._sensors:
            self.add_sensor(name)

        sensor = self._sensors[name]
        sensor['history'].append(float(value))
        sensor['timestamps'].append(time.time())

        # Trim to window
        if len(sensor['history']) > self._window_size:
            sensor['history'] = sensor['history'][-self._window_size:]
            sensor['timestamps'] = sensor['timestamps'][-self._window_size:]

    def get_state(self) -> Dict[str, Any]:
        """
        Compute the current criticality state from all sensors.

        Returns:
            Dictionary with overall sigma_c and per-sensor details.
        """
        if not self._sensors:
            return {'sigma_c': 0.0, 'status': 'idle', 'sensors': {}}

        per_sensor = {}
        weighted_sigma_c = 0.0
        total_weight = 0.0

        for name, sensor in self._sensors.items():
            history = sensor['history']
            if len(history) < 10:
                per_sensor[name] = {'sigma_c': 0.0, 'n_readings': len(history)}
                continue

            # Use time index as epsilon, sensor values as observable
            eps = np.linspace(0, 1, len(history))
            obs = np.array(history, dtype=np.float64)

            result = self._engine.compute_susceptibility(eps, obs)
            sc = result['sigma_c']

            per_sensor[name] = {
                'sigma_c': sc,
                'kappa': result['kappa'],
                'current_value': history[-1],
                'n_readings': len(history),
            }

            weighted_sigma_c += sensor['weight'] * sc
            total_weight += sensor['weight']

        overall = weighted_sigma_c / total_weight if total_weight > 0 else 0.0

        status = 'normal'
        if overall > 0.8:
            status = 'critical'
        elif overall > 0.5:
            status = 'warning'

        return {
            'sigma_c': overall,
            'status': status,
            'sensors': per_sensor,
        }

    def generate_config(self) -> str:
        """
        Generate example Home Assistant configuration YAML.

        Returns:
            YAML string for configuration.yaml.
        """
        sensor_lines = []
        for name, sensor in self._sensors.items():
            sensor_lines.append(
                f"      - entity_id: sensor.{name}\n"
                f"        weight: {sensor['weight']}"
            )

        sensors_yaml = '\n'.join(sensor_lines)
        return f"""# configuration.yaml
sensor:
  - platform: sigma_c
    name: "System Criticality"
    monitors:
{sensors_yaml}
    scan_interval: 60

automation:
  - alias: "High Criticality Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.system_criticality
        above: 0.8
    action:
      - service: notify.mobile_app
        data:
          message: "System criticality high: sigma_c = {{{{ states('sensor.system_criticality') }}}}"
"""


class SigmaCSensor(SensorEntity):
    """
    Home Assistant sensor entity for criticality monitoring.

    Implements the HA SensorEntity protocol. Aggregates readings from
    other HA entities and exposes a combined criticality score.

    Requires Home Assistant to be installed.
    """

    def __init__(self, name: str, monitors: List[Dict[str, Any]],
                 scan_interval: int = 60):
        if not _HAS_HA:
            raise ImportError(
                "Home Assistant not installed. Use HomeAssistantBridge for "
                "standalone testing, or install homeassistant."
            )
        self._attr_name = name
        self._monitors = monitors
        self._state = 0.0
        self._bridge = HomeAssistantBridge(window_size=100)

        for monitor in monitors:
            self._bridge.add_sensor(
                monitor['entity_id'],
                weight=monitor.get('weight', 1.0),
            )

    @property
    def name(self) -> str:
        return self._attr_name

    @property
    def state(self) -> float:
        return round(self._state, 4)

    @property
    def unit_of_measurement(self) -> str:
        return "sigma_c"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Expose per-sensor details as HA attributes."""
        state = self._bridge.get_state()
        return {
            'status': state['status'],
            'sensors': state['sensors'],
        }

    def update(self):
        """Called by HA on each scan interval."""
        for monitor in self._monitors:
            entity_id = monitor['entity_id']
            ha_state = self.hass.states.get(entity_id)

            if ha_state and ha_state.state not in ('unknown', 'unavailable'):
                try:
                    value = float(ha_state.state)
                    self._bridge.update(entity_id, value)
                except ValueError:
                    pass

        state = self._bridge.get_state()
        self._state = state['sigma_c']
