"""
Sigma-C Grafana / Prometheus Integration
==========================================
Copyright (c) 2025 ForgottenForge.xyz

Export Sigma-C criticality metrics to Prometheus for Grafana dashboards.
Supports both push (via gateway) and pull (via HTTP server) modes.

Requires: pip install prometheus-client

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import threading
import time
from typing import Dict, Any, Optional, Callable

try:
    from prometheus_client import (
        Gauge, CollectorRegistry, push_to_gateway, start_http_server
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class GrafanaExporter:
    """
    Export Sigma-C metrics to Prometheus/Grafana.

    Two modes of operation:
    1. Push mode: Periodically pushes metrics to a Prometheus Pushgateway.
    2. Pull mode: Starts an HTTP server that Prometheus scrapes.

    Usage (push mode):
        from sigma_c.monitoring.grafana import GrafanaExporter

        exporter = GrafanaExporter(gateway='localhost:9091')
        exporter.push_metrics('quantum_system', sigma_c=0.42, kappa=12.3, chi_max=0.85)

    Usage (pull mode):
        exporter = GrafanaExporter()
        exporter.start_http_server(port=8000)
        exporter.update_metrics('gpu_cluster', sigma_c=0.71, kappa=8.5, chi_max=1.2)

    Usage (streaming):
        exporter = GrafanaExporter(gateway='localhost:9091')
        exporter.start_streaming(
            compute_fn=lambda: {'sigma_c': 0.5, 'kappa': 3.0, 'chi_max': 0.8},
            system_name='my_system',
            interval=30
        )
    """

    def __init__(self, gateway: Optional[str] = None, job_name: str = 'sigma_c'):
        if not _HAS_PROMETHEUS:
            raise ImportError(
                "prometheus_client not installed. Run: pip install prometheus-client"
            )

        self._gateway = gateway
        self._job_name = job_name
        self._registry = CollectorRegistry()
        self._streaming_thread = None
        self._stop_event = threading.Event()

        # Define gauges on our private registry
        self.sigma_c_gauge = Gauge(
            'sigma_c_value', 'Current criticality value',
            ['system'], registry=self._registry
        )
        self.kappa_gauge = Gauge(
            'sigma_c_kappa', 'Peak sharpness',
            ['system'], registry=self._registry
        )
        self.chi_max_gauge = Gauge(
            'sigma_c_chi_max', 'Maximum susceptibility',
            ['system'], registry=self._registry
        )

    def update_metrics(self, system_name: str, sigma_c: float,
                       kappa: float, chi_max: float):
        """
        Update metric values (for pull mode or before push).

        Args:
            system_name: Label for the monitored system.
            sigma_c: Criticality value.
            kappa: Peak sharpness.
            chi_max: Maximum susceptibility.
        """
        self.sigma_c_gauge.labels(system=system_name).set(sigma_c)
        self.kappa_gauge.labels(system=system_name).set(kappa)
        self.chi_max_gauge.labels(system=system_name).set(chi_max)

    def push_metrics(self, system_name: str, sigma_c: float,
                     kappa: float, chi_max: float):
        """
        Push metrics to a Prometheus Pushgateway.

        Args:
            system_name: Label for the monitored system.
            sigma_c: Criticality value.
            kappa: Peak sharpness.
            chi_max: Maximum susceptibility.
        """
        if not self._gateway:
            raise ValueError(
                "No gateway configured. Pass gateway= to constructor."
            )

        self.update_metrics(system_name, sigma_c, kappa, chi_max)
        push_to_gateway(
            self._gateway, job=self._job_name, registry=self._registry
        )

    def start_http_server(self, port: int = 8000):
        """
        Start HTTP server for Prometheus pull scraping.

        Args:
            port: Port to listen on.
        """
        start_http_server(port, registry=self._registry)

    def start_streaming(self, compute_fn: Callable[[], Dict[str, float]],
                        system_name: str, interval: int = 30):
        """
        Start a background thread that periodically computes and pushes metrics.

        Args:
            compute_fn: Callable returning {'sigma_c': ..., 'kappa': ..., 'chi_max': ...}.
            system_name: Label for the monitored system.
            interval: Update interval in seconds.
        """
        if not self._gateway:
            raise ValueError(
                "Streaming requires a gateway. Pass gateway= to constructor."
            )

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    metrics = compute_fn()
                    self.push_metrics(
                        system_name,
                        sigma_c=metrics.get('sigma_c', 0.0),
                        kappa=metrics.get('kappa', 0.0),
                        chi_max=metrics.get('chi_max', 0.0),
                    )
                except Exception:
                    pass
                self._stop_event.wait(interval)

        self._streaming_thread = threading.Thread(target=_loop, daemon=True)
        self._streaming_thread.start()

    def stop_streaming(self):
        """Stop the background streaming thread."""
        self._stop_event.set()
        if self._streaming_thread:
            self._streaming_thread.join(timeout=5)
            self._streaming_thread = None


# Grafana dashboard provisioning JSON
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Sigma-C Criticality Monitor",
        "panels": [
            {
                "title": "System Criticality (sigma_c)",
                "type": "timeseries",
                "targets": [{"expr": "sigma_c_value", "legendFormat": "{{system}}"}],
                "fieldConfig": {"defaults": {"min": 0, "max": 1}},
            },
            {
                "title": "Peak Sharpness (kappa)",
                "type": "timeseries",
                "targets": [{"expr": "sigma_c_kappa", "legendFormat": "{{system}}"}],
            },
            {
                "title": "Maximum Susceptibility (chi_max)",
                "type": "timeseries",
                "targets": [{"expr": "sigma_c_chi_max", "legendFormat": "{{system}}"}],
            },
        ],
    }
}
