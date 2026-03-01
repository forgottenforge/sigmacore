"""
Sigma-C GitHub Actions / CI Integration
=========================================
Copyright (c) 2025 ForgottenForge.xyz

CI/CD integration for automated code criticality analysis.
Analyzes Python source files for structural complexity metrics and
maps them onto a criticality score using Sigma-C conventions.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import ast
import os
import json
from typing import Dict, Any, List, Optional


class CIAnalyzer:
    """
    Analyze a Python codebase for structural criticality.

    Computes per-file and aggregate metrics using AST analysis:
    - Cyclomatic complexity (branching density)
    - Nesting depth
    - Function length distribution
    - Coupling (import count)

    These are combined into a single sigma_c score (0-1) where higher
    values indicate a codebase closer to a structural "phase transition"
    (becoming hard to maintain).

    Usage:
        from sigma_c.monitoring.github_actions import CIAnalyzer

        analyzer = CIAnalyzer(threshold=0.7)
        report = analyzer.analyze_repository('src/')
        print(f"sigma_c = {report['sigma_c']:.3f}")

        if report['status'] == 'critical':
            print("Codebase complexity is above threshold!")
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze a single Python file for complexity metrics.

        Args:
            filepath: Path to a .py file.

        Returns:
            Dictionary with complexity metrics and sigma_c score.
        """
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            return {
                'file': filepath,
                'error': 'SyntaxError',
                'sigma_c': 0.0,
                'metrics': {},
            }

        metrics = self._extract_metrics(tree, source)
        sigma_c = self._compute_file_criticality(metrics)

        return {
            'file': filepath,
            'sigma_c': sigma_c,
            'metrics': metrics,
        }

    def analyze_repository(self, path: str = '.') -> Dict[str, Any]:
        """
        Analyze all Python files in a directory tree.

        Args:
            path: Root directory to scan.

        Returns:
            Aggregate report with per-file results and overall sigma_c.
        """
        file_results = []

        for root, _dirs, files in os.walk(path):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                filepath = os.path.join(root, fname)
                result = self.analyze_file(filepath)
                file_results.append(result)

        if not file_results:
            return {
                'sigma_c': 0.0,
                'status': 'pass',
                'n_files': 0,
                'files': [],
                'threshold': self.threshold,
            }

        scores = [r['sigma_c'] for r in file_results if 'error' not in r]
        overall_sigma_c = max(scores) if scores else 0.0

        status = 'critical' if overall_sigma_c > self.threshold else 'pass'

        return {
            'sigma_c': overall_sigma_c,
            'mean_sigma_c': sum(scores) / len(scores) if scores else 0.0,
            'max_sigma_c': overall_sigma_c,
            'status': status,
            'n_files': len(file_results),
            'critical_files': [
                r for r in file_results if r['sigma_c'] > self.threshold
            ],
            'files': file_results,
            'threshold': self.threshold,
        }

    def generate_report(self, result: Dict[str, Any], fmt: str = 'json') -> str:
        """
        Generate a human-readable or machine-readable report.

        Args:
            result: Output from analyze_repository().
            fmt: 'json' or 'text'.

        Returns:
            Formatted report string.
        """
        if fmt == 'json':
            return json.dumps(result, indent=2, default=str)

        lines = [
            "=" * 60,
            "  Sigma-C CI Criticality Report",
            "=" * 60,
            f"  Files analyzed:  {result['n_files']}",
            f"  Overall sigma_c: {result['sigma_c']:.3f}",
            f"  Mean sigma_c:    {result.get('mean_sigma_c', 0):.3f}",
            f"  Threshold:       {result['threshold']:.3f}",
            f"  Status:          {result['status'].upper()}",
        ]

        critical = result.get('critical_files', [])
        if critical:
            lines.append("")
            lines.append("  Critical files:")
            for f in critical:
                lines.append(f"    {f['file']}: sigma_c = {f['sigma_c']:.3f}")

        lines.append("=" * 60)
        return '\n'.join(lines)

    def _extract_metrics(self, tree: ast.AST, source: str) -> Dict[str, Any]:
        """Extract complexity metrics from an AST."""
        lines = source.split('\n')

        functions = []
        classes = []
        imports = 0
        max_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                complexity = self._cyclomatic_complexity(node)
                depth = self._max_nesting_depth(node)
                functions.append({
                    'name': node.name,
                    'lines': func_lines,
                    'complexity': complexity,
                    'depth': depth,
                })
                max_depth = max(max_depth, depth)

            elif isinstance(node, ast.ClassDef):
                class_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                n_methods = sum(
                    1 for child in ast.walk(node)
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                classes.append({
                    'name': node.name,
                    'lines': class_lines,
                    'n_methods': n_methods,
                })

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1

        total_lines = len(lines)
        non_empty = sum(1 for l in lines if l.strip())

        return {
            'total_lines': total_lines,
            'non_empty_lines': non_empty,
            'n_functions': len(functions),
            'n_classes': len(classes),
            'n_imports': imports,
            'max_nesting_depth': max_depth,
            'functions': functions,
            'classes': classes,
            'avg_function_length': (
                sum(f['lines'] for f in functions) / len(functions)
                if functions else 0
            ),
            'avg_complexity': (
                sum(f['complexity'] for f in functions) / len(functions)
                if functions else 0
            ),
            'max_complexity': (
                max(f['complexity'] for f in functions)
                if functions else 0
            ),
        }

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Compute cyclomatic complexity of an AST node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Assert, ast.With, ast.AsyncWith)):
                complexity += 1
        return complexity

    def _max_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Compute maximum nesting depth of an AST node."""
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                  ast.With, ast.AsyncWith, ast.Try)):
                child_depth = self._max_nesting_depth(child, depth + 1)
                max_d = max(max_d, child_depth)
            else:
                child_depth = self._max_nesting_depth(child, depth)
                max_d = max(max_d, child_depth)
        return max_d

    def _compute_file_criticality(self, metrics: Dict[str, Any]) -> float:
        """
        Map complexity metrics to a criticality score in [0, 1].

        Weights:
        - 40% cyclomatic complexity (normalized to 0-1, cap at 20)
        - 25% max nesting depth (normalized, cap at 8)
        - 20% average function length (normalized, cap at 100 lines)
        - 15% coupling (import count, normalized, cap at 30)
        """
        cc = min(metrics.get('max_complexity', 0) / 20.0, 1.0)
        depth = min(metrics.get('max_nesting_depth', 0) / 8.0, 1.0)
        func_len = min(metrics.get('avg_function_length', 0) / 100.0, 1.0)
        coupling = min(metrics.get('n_imports', 0) / 30.0, 1.0)

        sigma_c = 0.40 * cc + 0.25 * depth + 0.20 * func_len + 0.15 * coupling
        return round(sigma_c, 4)


def generate_action_yaml() -> str:
    """Generate the GitHub Action metadata YAML."""
    return """name: Sigma-C Criticality Check
description: 'Analyze code criticality with Sigma-C'
author: 'ForgottenForge'

inputs:
  threshold:
    description: 'Maximum allowed sigma_c value'
    required: false
    default: '0.7'
  fail-on-critical:
    description: 'Fail build if threshold exceeded'
    required: false
    default: 'true'
  path:
    description: 'Path to analyze'
    required: false
    default: '.'

outputs:
  sigma_c:
    description: 'Computed criticality value'
  status:
    description: 'Pass or fail status'

runs:
  using: 'composite'
  steps:
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install Sigma-C
      shell: bash
      run: pip install sigma-c-framework

    - name: Run Analysis
      shell: bash
      run: |
        python -c "
        from sigma_c.monitoring.github_actions import CIAnalyzer
        import json

        analyzer = CIAnalyzer(threshold=float('${{ inputs.threshold }}'))
        result = analyzer.analyze_repository('${{ inputs.path }}')

        print(f'sigma_c={result[\"sigma_c\"]:.4f}')
        print(f'status={result[\"status\"]}')

        with open('sigma_c_report.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)

        if result['status'] == 'critical' and '${{ inputs.fail-on-critical }}' == 'true':
            exit(1)
        "

branding:
  icon: 'activity'
  color: 'blue'
"""


def generate_workflow_yaml() -> str:
    """Generate an example GitHub Actions workflow YAML."""
    return """# .github/workflows/criticality.yml
name: Criticality Check
on: [push, pull_request]

jobs:
  sigma-c:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Sigma-C
        run: pip install sigma-c-framework

      - name: Analyze Criticality
        run: |
          python -c "
          from sigma_c.monitoring.github_actions import CIAnalyzer
          analyzer = CIAnalyzer(threshold=0.7)
          result = analyzer.analyze_repository('.')
          report = analyzer.generate_report(result, fmt='text')
          print(report)
          if result['status'] == 'critical':
              exit(1)
          "

      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: criticality-report
          path: sigma_c_report.json
"""


def analyze_repository(path: str = '.', threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function for CI usage.

    Args:
        path: Directory to analyze.
        threshold: Criticality threshold.

    Returns:
        Analysis result dictionary.
    """
    analyzer = CIAnalyzer(threshold=threshold)
    return analyzer.analyze_repository(path)
