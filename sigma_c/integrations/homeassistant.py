"""
Sigma-C Home Assistant Sensor
==============================
Copyright (c) 2025 ForgottenForge.xyz

Home Assistant custom component for system criticality monitoring.
"""

# This would be placed in custom_components/sigma_c/

MANIFEST = """
{
  "domain": "sigma_c",
  "name": "Sigma-C Criticality Monitor",
  "documentation": "https://sigma-c.io/integrations/home-assistant",
  "requirements": ["sigma-c-framework>=2.0.0"],
  "codeowners": ["@forgottenforge"],
  "version": "2.0.0"
}
"""

SENSOR_CONFIG = """
# configuration.yaml

sensor:
  - platform: sigma_c
    name: "House Criticality"
    monitors:
      - entity_id: sensor.energy_usage
        weight: 0.3
      - entity_id: sensor.temperature
        weight: 0.3
      - entity_id: sensor.network_traffic
        weight: 0.4
    scan_interval: 60
    
automation:
  - alias: "High Criticality Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.house_criticality
        above: 0.8
    action:
      - service: notify.mobile_app
        data:
          message: "System criticality high! σ_c = {{ states('sensor.house_criticality') }}"
"""

SENSOR_CODE = """
# custom_components/sigma_c/sensor.py

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
import numpy as np

from sigma_c.core.engine import Engine

class SigmaCSensor(SensorEntity):
    '''Sigma-C criticality sensor.'''
    
    def __init__(self, name, monitors, scan_interval):
        self._name = name
        self._monitors = monitors
        self._state = 0.0
        self._engine = Engine()
        self._history = {m['entity_id']: [] for m in monitors}
    
    @property
    def name(self):
        return self._name
    
    @property
    def state(self):
        return self._state
    
    @property
    def unit_of_measurement(self):
        return "σ_c"
    
    def update(self):
        '''Update sensor state.'''
        # Collect current values
        values = []
        for monitor in self._monitors:
            entity_id = monitor['entity_id']
            state = self.hass.states.get(entity_id)
            
            if state and state.state not in ['unknown', 'unavailable']:
                try:
                    value = float(state.state)
                    self._history[entity_id].append(value)
                    
                    # Keep only recent history
                    if len(self._history[entity_id]) > 100:
                        self._history[entity_id] = self._history[entity_id][-100:]
                    
                    values.append(value)
                except ValueError:
                    pass
        
        # Compute criticality
        if values:
            # Simple metric: variance of recent values
            sigma_c = np.std(values) / (np.mean(values) + 1e-9)
            self._state = float(np.clip(sigma_c, 0, 1))
"""
