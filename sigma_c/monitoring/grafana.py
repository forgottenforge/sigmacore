"""
Sigma-C Grafana Plugin
======================
Copyright (c) 2025 ForgottenForge.xyz

Grafana data source plugin for real-time criticality monitoring.
"""

from typing import Dict, Any, List
import time

try:
    from prometheus_client import Gauge, push_to_gateway
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class GrafanaExporter:
    """
    Export Sigma-C metrics to Prometheus/Grafana.
    
    Usage:
        from sigma_c.monitoring.grafana import GrafanaExporter
        
        exporter = GrafanaExporter(
            prometheus_gateway='localhost:9091'
        )
        exporter.start_streaming()
    """
    
    def __init__(self, prometheus_gateway: str = 'localhost:9091'):
        if not _HAS_PROMETHEUS:
            raise ImportError("prometheus_client not installed. Run: pip install prometheus-client")
        
        self.gateway = prometheus_gateway
        
        # Define metrics
        self.sigma_c_gauge = Gauge('sigma_c_value', 'Current criticality value', ['system'])
        self.kappa_gauge = Gauge('sigma_c_kappa', 'Peak sharpness', ['system'])
        self.chi_max_gauge = Gauge('sigma_c_chi_max', 'Maximum susceptibility', ['system'])
    
    def push_metrics(self, system_name: str, sigma_c: float, kappa: float, chi_max: float):
        """
        Push metrics to Prometheus gateway.
        
        Args:
            system_name: Name of the system being monitored
            sigma_c: Criticality value
            kappa: Peak sharpness
            chi_max: Maximum susceptibility
        """
        self.sigma_c_gauge.labels(system=system_name).set(sigma_c)
        self.kappa_gauge.labels(system=system_name).set(kappa)
        self.chi_max_gauge.labels(system=system_name).set(chi_max)
        
        push_to_gateway(self.gateway, job='sigma_c', registry=None)
    
    def start_streaming(self, interval: int = 30):
        """
        Start continuous metric streaming.
        
        Args:
            interval: Update interval in seconds
        """
        print(f"Starting Grafana streaming (interval={interval}s)")
        print(f"Metrics available at: http://{self.gateway}/metrics")
        
        # In production, this would run in a background thread
        # For now, just a stub
        pass


# Grafana Dashboard JSON (example)
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Sigma-C Criticality Monitor",
        "panels": [
            {
                "title": "System Criticality",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sigma_c_value",
                        "legendFormat": "{{system}}"
                    }
                ],
                "yaxes": [
                    {"label": "σ_c", "min": 0, "max": 1}
                ]
            },
            {
                "title": "Peak Sharpness (κ)",
                "type": "graph",
                "targets": [
                    {
                        "expr": "sigma_c_kappa",
                        "legendFormat": "{{system}}"
                    }
                ]
            },
            {
                "title": "Criticality Heatmap",
                "type": "heatmap",
                "targets": [
                    {
                        "expr": "sigma_c_value",
                        "format": "time_series"
                    }
                ]
            }
        ]
    }
}
