"""
Sigma-C Kubernetes Monitoring
===============================
Copyright (c) 2025 ForgottenForge.xyz

Kubernetes integration for criticality-based pod monitoring and autoscaling.
Collects resource metrics from pods, computes criticality scores, and
optionally triggers scaling actions when thresholds are exceeded.

Requires: pip install kubernetes

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from kubernetes import client, config, watch
    _HAS_K8S = True
except ImportError:
    _HAS_K8S = False


# CRD definition for reference (apply with kubectl apply -f)
CRD_YAML = """
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: criticalitymonitors.sigma-c.io
spec:
  group: sigma-c.io
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                target:
                  type: object
                  properties:
                    deployment:
                      type: string
                    namespace:
                      type: string
                thresholds:
                  type: object
                  properties:
                    sigma_c:
                      type: number
                interval:
                  type: integer
  scope: Namespaced
  names:
    plural: criticalitymonitors
    singular: criticalitymonitor
    kind: CriticalityMonitor
    shortNames:
    - scm
"""


class KubernetesMonitor:
    """
    Monitor Kubernetes pods and compute criticality from resource usage.

    Collects CPU and memory utilization over a sliding window, then
    computes sigma_c from the susceptibility of resource usage patterns.
    High sigma_c indicates the deployment is near a resource saturation
    transition.

    Usage:
        from sigma_c.monitoring.kubernetes import KubernetesMonitor

        monitor = KubernetesMonitor()
        monitor.connect()

        metrics = monitor.collect_metrics('my-app', namespace='default')
        sigma_c = monitor.compute_criticality(metrics)
        print(f"Deployment criticality: {sigma_c:.3f}")

        # Autoscale if critical
        if sigma_c > 0.8:
            monitor.scale_deployment('my-app', replicas=5, namespace='default')
    """

    def __init__(self, kubeconfig: Optional[str] = None):
        if not _HAS_K8S:
            raise ImportError(
                "kubernetes client not installed. Run: pip install kubernetes"
            )
        self._kubeconfig = kubeconfig
        self._v1 = None
        self._apps_v1 = None
        self._metrics_api = None
        self._history: Dict[str, List[Dict[str, float]]] = {}

    def connect(self, in_cluster: bool = False):
        """
        Connect to a Kubernetes cluster.

        Args:
            in_cluster: If True, use in-cluster config (for pods).
                        If False, use kubeconfig file.
        """
        if in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config(config_file=self._kubeconfig)

        self._v1 = client.CoreV1Api()
        self._apps_v1 = client.AppsV1Api()
        self._metrics_api = client.CustomObjectsApi()

    def collect_metrics(self, deployment_name: str,
                        namespace: str = 'default') -> Dict[str, Any]:
        """
        Collect resource metrics for all pods in a deployment.

        Args:
            deployment_name: Name of the Kubernetes deployment.
            namespace: Kubernetes namespace.

        Returns:
            Dictionary with per-pod CPU and memory metrics.
        """
        label_selector = f"app={deployment_name}"
        pods = self._v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
        )

        pod_metrics = []
        try:
            metrics_response = self._metrics_api.list_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods",
            )
            metrics_by_name = {
                m['metadata']['name']: m
                for m in metrics_response.get('items', [])
            }
        except Exception:
            metrics_by_name = {}

        for pod in pods.items:
            pod_name = pod.metadata.name
            pod_phase = pod.status.phase

            cpu_usage = 0.0
            memory_usage = 0.0

            if pod_name in metrics_by_name:
                containers = metrics_by_name[pod_name].get('containers', [])
                for c in containers:
                    cpu_str = c.get('usage', {}).get('cpu', '0')
                    mem_str = c.get('usage', {}).get('memory', '0')
                    cpu_usage += self._parse_cpu(cpu_str)
                    memory_usage += self._parse_memory(mem_str)

            pod_metrics.append({
                'name': pod_name,
                'phase': pod_phase,
                'cpu_cores': cpu_usage,
                'memory_mb': memory_usage,
            })

        result = {
            'deployment': deployment_name,
            'namespace': namespace,
            'n_pods': len(pod_metrics),
            'pods': pod_metrics,
            'timestamp': time.time(),
        }

        # Store in history for time-series analysis
        key = f"{namespace}/{deployment_name}"
        if key not in self._history:
            self._history[key] = []
        self._history[key].append({
            'timestamp': result['timestamp'],
            'mean_cpu': np.mean([p['cpu_cores'] for p in pod_metrics]) if pod_metrics else 0,
            'mean_memory': np.mean([p['memory_mb'] for p in pod_metrics]) if pod_metrics else 0,
            'n_pods': len(pod_metrics),
        })
        # Keep last 100 measurements
        if len(self._history[key]) > 100:
            self._history[key] = self._history[key][-100:]

        return result

    def compute_criticality(self, metrics: Optional[Dict[str, Any]] = None,
                            deployment_name: Optional[str] = None,
                            namespace: str = 'default') -> float:
        """
        Compute criticality score from resource usage history.

        Uses the susceptibility (derivative) of CPU/memory usage over time.
        A sharp peak in the derivative indicates the system is near a
        resource saturation transition.

        Args:
            metrics: Output from collect_metrics() (uses its deployment key).
            deployment_name: Alternatively, specify deployment directly.
            namespace: Kubernetes namespace.

        Returns:
            Criticality score in [0, 1].
        """
        if metrics is not None:
            key = f"{metrics['namespace']}/{metrics['deployment']}"
        elif deployment_name is not None:
            key = f"{namespace}/{deployment_name}"
        else:
            return 0.0

        history = self._history.get(key, [])
        if len(history) < 3:
            return 0.0

        cpu_values = np.array([h['mean_cpu'] for h in history])
        mem_values = np.array([h['mean_memory'] for h in history])

        # Susceptibility: derivative of resource usage over time
        cpu_chi = np.abs(np.diff(cpu_values))
        mem_chi = np.abs(np.diff(mem_values))

        cpu_kappa = 0.0
        if np.mean(cpu_chi) > 1e-9:
            cpu_kappa = float(np.max(cpu_chi) / np.mean(cpu_chi))

        mem_kappa = 0.0
        if np.mean(mem_chi) > 1e-9:
            mem_kappa = float(np.max(mem_chi) / np.mean(mem_chi))

        # Combined criticality (normalize kappa to [0, 1])
        combined_kappa = max(cpu_kappa, mem_kappa)
        sigma_c = float(np.clip(1.0 - 1.0 / (1.0 + combined_kappa), 0, 1))
        return sigma_c

    def scale_deployment(self, deployment_name: str, replicas: int,
                         namespace: str = 'default'):
        """
        Scale a deployment to the specified number of replicas.

        Args:
            deployment_name: Name of the deployment.
            replicas: Target replica count.
            namespace: Kubernetes namespace.
        """
        body = {"spec": {"replicas": replicas}}
        self._apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body,
        )

    def watch_pods(self, namespace: str = 'default', timeout: int = 60):
        """
        Watch pod events in a namespace (generator).

        Args:
            namespace: Kubernetes namespace to watch.
            timeout: Watch timeout in seconds.

        Yields:
            Event dictionaries with type and pod info.
        """
        w = watch.Watch()
        for event in w.stream(
            self._v1.list_namespaced_pod,
            namespace=namespace,
            timeout_seconds=timeout,
        ):
            yield {
                'type': event['type'],
                'pod': event['object'].metadata.name,
                'phase': event['object'].status.phase,
                'namespace': namespace,
            }

    def get_deployment_status(self, deployment_name: str,
                              namespace: str = 'default') -> Dict[str, Any]:
        """
        Get current status of a deployment.

        Args:
            deployment_name: Name of the deployment.
            namespace: Kubernetes namespace.

        Returns:
            Dictionary with replicas, ready count, and conditions.
        """
        dep = self._apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
        )
        return {
            'name': deployment_name,
            'namespace': namespace,
            'replicas': dep.spec.replicas,
            'ready_replicas': dep.status.ready_replicas or 0,
            'available_replicas': dep.status.available_replicas or 0,
            'conditions': [
                {'type': c.type, 'status': c.status, 'reason': c.reason}
                for c in (dep.status.conditions or [])
            ],
        }

    @staticmethod
    def _parse_cpu(cpu_str: str) -> float:
        """Parse Kubernetes CPU string (e.g. '250m', '1') to cores."""
        if cpu_str.endswith('n'):
            return float(cpu_str[:-1]) / 1e9
        elif cpu_str.endswith('u'):
            return float(cpu_str[:-1]) / 1e6
        elif cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000.0
        else:
            return float(cpu_str)

    @staticmethod
    def _parse_memory(mem_str: str) -> float:
        """Parse Kubernetes memory string (e.g. '128Mi', '1Gi') to MB."""
        if mem_str.endswith('Ki'):
            return float(mem_str[:-2]) / 1024.0
        elif mem_str.endswith('Mi'):
            return float(mem_str[:-2])
        elif mem_str.endswith('Gi'):
            return float(mem_str[:-2]) * 1024.0
        elif mem_str.endswith('Ti'):
            return float(mem_str[:-2]) * 1024.0 * 1024.0
        elif mem_str.endswith('k'):
            return float(mem_str[:-1]) / 1000.0
        elif mem_str.endswith('M'):
            return float(mem_str[:-1])
        elif mem_str.endswith('G'):
            return float(mem_str[:-1]) * 1000.0
        else:
            return float(mem_str) / (1024.0 * 1024.0)  # bytes to MB
