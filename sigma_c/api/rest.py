"""
Sigma-C REST API
================
Copyright (c) 2025 ForgottenForge.xyz

FastAPI-based REST API for Sigma-C analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    BaseModel = object

from ..core.engine import Engine


class AnalysisRequest(BaseModel if _HAS_FASTAPI else object):
    """Request model for analysis endpoint."""
    epsilon: List[float]
    observable: List[float]
    method: Optional[str] = "fft"


class AnalysisResponse(BaseModel if _HAS_FASTAPI else object):
    """Response model for analysis endpoint."""
    sigma_c: float
    kappa: float
    chi_max: float
    peak_location: float


class SigmaCAPI:
    """
    REST API wrapper for Sigma-C framework.
    
    Usage:
        from fastapi import FastAPI
        from sigma_c.api import SigmaCAPI
        
        app = FastAPI()
        sigma_api = SigmaCAPI()
        
        @app.post("/analyze")
        async def analyze(data: List[float]):
            return {"sigma_c": sigma_api.compute(data)}
    """
    
    def __init__(self):
        self.engine = Engine()
    
    def compute(self, epsilon: List[float], observable: List[float]) -> Dict[str, float]:
        """
        Compute criticality from data.
        
        Args:
            epsilon: Control parameter values
            observable: Observable values
            
        Returns:
            Dictionary with sigma_c and related metrics
        """
        eps_array = np.array(epsilon)
        obs_array = np.array(observable)
        
        result = self.engine.compute_susceptibility(eps_array, obs_array)
        
        return {
            'sigma_c': float(result['sigma_c']),
            'kappa': float(result['kappa']),
            'chi_max': float(result['chi_max']),
            'peak_location': float(result.get('peak_location', result['sigma_c']))
        }
    
    def create_app(self) -> Any:
        """
        Create a FastAPI application with all endpoints.
        
        Returns:
            FastAPI app instance
        """
        if not _HAS_FASTAPI:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
        
        app = FastAPI(
            title="Sigma-C API",
            description="Critical susceptibility analysis API",
            version="2.0.0"
        )
        
        @app.post("/analyze", response_model=AnalysisResponse)
        async def analyze(request: AnalysisRequest):
            """
            Analyze criticality from epsilon-observable data.
            """
            try:
                result = self.compute(request.epsilon, request.observable)
                return AnalysisResponse(**result)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "version": "2.0.0"}
        
        return app


# GraphQL Schema (as string for documentation)
GRAPHQL_SCHEMA = """
type CriticalityAnalysis {
  sigmaC: Float!
  kappa: Float!
  chiMax: Float!
  peakLocation: Float!
  confidence: Float!
}

type Peak {
  location: Float!
  height: Float!
  width: Float!
}

input AnalysisInput {
  epsilon: [Float!]!
  observable: [Float!]!
  method: String
}

type Query {
  analyzeSystem(input: AnalysisInput!): CriticalityAnalysis!
  health: String!
}
"""
