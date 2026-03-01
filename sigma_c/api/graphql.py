"""
Sigma-C GraphQL API
====================
Copyright (c) 2025 ForgottenForge.xyz

GraphQL endpoint for criticality analysis.
Uses strawberry-graphql when available, provides a minimal built-in
resolver when it is not.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import re
import json
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.engine import Engine

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    _HAS_STRAWBERRY = True
except ImportError:
    _HAS_STRAWBERRY = False


# ── Strawberry-based GraphQL (when installed) ────────────────────────

def create_strawberry_schema():
    """
    Create a strawberry GraphQL schema with full type safety.

    Returns:
        strawberry.Schema instance.

    Raises:
        ImportError: If strawberry-graphql is not installed.
    """
    if not _HAS_STRAWBERRY:
        raise ImportError(
            "strawberry-graphql not installed. Run: pip install strawberry-graphql"
        )

    @strawberry.type
    class CriticalityResult:
        sigma_c: float
        kappa: float
        chi_max: float
        peak_location: float

    @strawberry.input
    class AnalysisInput:
        epsilon: List[float]
        observable: List[float]

    @strawberry.type
    class HealthStatus:
        status: str
        version: str

    engine = Engine()

    @strawberry.type
    class Query:
        @strawberry.field
        def analyze_system(self, input: AnalysisInput) -> CriticalityResult:
            eps = np.array(input.epsilon, dtype=np.float64)
            obs = np.array(input.observable, dtype=np.float64)
            result = engine.compute_susceptibility(eps, obs)
            return CriticalityResult(
                sigma_c=float(result['sigma_c']),
                kappa=float(result['kappa']),
                chi_max=float(result['chi_max']),
                peak_location=float(result['sigma_c']),
            )

        @strawberry.field
        def health(self) -> HealthStatus:
            return HealthStatus(status="healthy", version="2.1.0")

    return strawberry.Schema(query=Query)


def create_graphql_router():
    """
    Create a FastAPI-compatible GraphQL router.

    Usage:
        from fastapi import FastAPI
        from sigma_c.api.graphql import create_graphql_router

        app = FastAPI()
        app.include_router(create_graphql_router(), prefix="/graphql")

    Returns:
        strawberry.fastapi.GraphQLRouter
    """
    schema = create_strawberry_schema()
    return GraphQLRouter(schema)


# ── Built-in minimal GraphQL resolver (no dependencies) ─────────────

class GraphQLAPI:
    """
    Minimal GraphQL API that works without any external dependencies.

    Parses a subset of GraphQL queries and resolves them against the
    Sigma-C engine. Supports the same schema as the strawberry version.

    Usage:
        from sigma_c.api.graphql import GraphQLAPI

        api = GraphQLAPI()

        # Execute a query
        result = api.execute('''
            {
                analyzeSystem(input: {
                    epsilon: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    observable: [1.0, 0.95, 0.8, 0.4, 0.1, 0.05]
                }) {
                    sigmaC
                    kappa
                    chiMax
                }
            }
        ''')
        print(result)
    """

    SCHEMA = """
type CriticalityResult {
  sigmaC: Float!
  kappa: Float!
  chiMax: Float!
  peakLocation: Float!
}

type HealthStatus {
  status: String!
  version: String!
}

input AnalysisInput {
  epsilon: [Float!]!
  observable: [Float!]!
}

type Query {
  analyzeSystem(input: AnalysisInput!): CriticalityResult!
  health: HealthStatus!
}
"""

    def __init__(self):
        self._engine = Engine()

    def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query string.

        Args:
            query: GraphQL query string.
            variables: Optional variables dict.

        Returns:
            Dictionary with 'data' key containing results.
        """
        query = query.strip()

        # Strip outer braces or query keyword
        if query.startswith('{'):
            query = query[1:].rsplit('}', 1)[0].strip()
        elif query.startswith('query'):
            # Remove "query { ... }"
            brace_start = query.index('{')
            query = query[brace_start + 1:].rsplit('}', 1)[0].strip()

        data = {}

        if 'analyzeSystem' in query:
            data['analyzeSystem'] = self._resolve_analyze(query, variables)

        if 'health' in query:
            data['health'] = self._resolve_health()

        return {'data': data}

    def _resolve_analyze(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve the analyzeSystem query."""
        # Extract epsilon and observable arrays from the query
        epsilon = self._extract_float_array(query, 'epsilon')
        observable = self._extract_float_array(query, 'observable')

        # Try variables if inline extraction failed
        if variables and (not epsilon or not observable):
            inp = variables.get('input', {})
            epsilon = epsilon or inp.get('epsilon', [])
            observable = observable or inp.get('observable', [])

        if not epsilon or not observable:
            return {'error': 'Missing epsilon or observable arrays'}

        eps = np.array(epsilon, dtype=np.float64)
        obs = np.array(observable, dtype=np.float64)
        result = self._engine.compute_susceptibility(eps, obs)

        full_result = {
            'sigmaC': float(result['sigma_c']),
            'kappa': float(result['kappa']),
            'chiMax': float(result['chi_max']),
            'peakLocation': float(result['sigma_c']),
        }

        # Filter to requested fields
        requested = self._extract_requested_fields(query, 'analyzeSystem')
        if requested:
            return {k: v for k, v in full_result.items() if k in requested}
        return full_result

    def _resolve_health(self) -> Dict[str, Any]:
        """Resolve the health query."""
        return {'status': 'healthy', 'version': '2.1.0'}

    @staticmethod
    def _extract_float_array(query: str, field_name: str) -> List[float]:
        """Extract a float array from a GraphQL query string."""
        pattern = field_name + r'\s*:\s*\[([^\]]*)\]'
        match = re.search(pattern, query)
        if not match:
            return []
        values_str = match.group(1)
        try:
            return [float(v.strip()) for v in values_str.split(',') if v.strip()]
        except ValueError:
            return []

    @staticmethod
    def _extract_requested_fields(query: str, operation: str) -> List[str]:
        """Extract the requested field names from a query."""
        # Find the operation, then skip past its arguments (handling nested braces)
        idx = query.find(operation)
        if idx < 0:
            return []
        idx += len(operation)

        # Skip past arguments by finding matching ) for the (
        paren_pos = query.find('(', idx)
        if paren_pos >= 0:
            depth = 0
            for i in range(paren_pos, len(query)):
                if query[i] == '(':
                    depth += 1
                elif query[i] == ')':
                    depth -= 1
                    if depth == 0:
                        idx = i + 1
                        break

        # Now find the next { ... } block which contains the result fields
        brace_start = query.find('{', idx)
        if brace_start < 0:
            return []
        brace_end = query.find('}', brace_start)
        if brace_end < 0:
            return []

        fields_str = query[brace_start + 1:brace_end]
        return [f.strip() for f in fields_str.split() if f.strip()]

    def get_schema(self) -> str:
        """Return the GraphQL schema definition."""
        return self.SCHEMA
