"""Sigma-C API - REST and GraphQL"""

try:
    from .rest import SigmaCAPI, GRAPHQL_SCHEMA
    __all__ = ['SigmaCAPI', 'GRAPHQL_SCHEMA']
except ImportError:
    __all__ = []
