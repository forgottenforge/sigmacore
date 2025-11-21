"""
Sigma-C Discovery Module
========================
Copyright (c) 2025 ForgottenForge.xyz

Implements rigorous methods for automatic observable discovery and 
multi-scale susceptibility analysis as defined in the reference papers.
"""

import numpy as np
from scipy import signal, stats
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ObservableCandidate:
    name: str
    score: float
    data: np.ndarray
    method: str

class ObservableDiscovery:
    """
    Automatically identifies the optimal order parameter (observable) 
    that maximizes the visibility of phase transitions.
    """
    
    def __init__(self, method: str = 'hybrid'):
        """
        Args:
            method: 'gradient', 'entropy', 'pca', or 'hybrid'
        """
        self.method = method

    def find_optimal_observable(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> ObservableCandidate:
        """
        Finds the best observable from a dataset of potential features.
        
        Args:
            data: Shape (n_samples, n_features)
            feature_names: Optional list of feature names
            
        Returns:
            ObservableCandidate with the best found observable
        """
        if data.ndim == 1:
            return ObservableCandidate("single_feature", 1.0, data, "identity")
            
        n_features = data.shape[1]
        candidates = []
        
        # 1. Gradient-Based Discovery (Maximal Susceptibility)
        # Look for features with highest variance in their gradients (peaks in susceptibility)
        for i in range(n_features):
            feat = data[:, i]
            # Calculate susceptibility chi = |dO/dx| (approximate via finite diff)
            chi = np.abs(np.gradient(feat))
            # Score is the peak-to-mean ratio of susceptibility (sharpness of transition)
            score = np.max(chi) / (np.mean(chi) + 1e-9)
            name = feature_names[i] if feature_names else f"feature_{i}"
            candidates.append(ObservableCandidate(name, score, feat, "gradient"))
            
        # 2. Entropy-Based Discovery (Information Theoretic)
        # Look for features that maximize information change
        for i in range(n_features):
            feat = data[:, i]
            # Discretize and calculate entropy
            hist, _ = np.histogram(feat, bins='auto', density=True)
            entropy = stats.entropy(hist + 1e-9)
            # In critical systems, entropy often peaks or changes sharply
            # We use a heuristic score here
            score = entropy 
            name = feature_names[i] if feature_names else f"feature_{i}"
            # Weight entropy score to be comparable
            candidates.append(ObservableCandidate(name, score, feat, "entropy"))

        # 3. PCA-Based (Collective Modes)
        # The principal component often captures the order parameter
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(data).flatten()
            explained_var = pca.explained_variance_ratio_[0]
            candidates.append(ObservableCandidate("PC1", explained_var * 10.0, pc1, "pca"))
        except ImportError:
            pass

        # Sort by score and return best
        best = max(candidates, key=lambda x: x.score)
        return best

class MultiScaleAnalysis:
    """
    Performs multi-resolution analysis to detect criticality across different scales.
    Essential for systems with hierarchical structure (e.g., GPU caches, turbulence).
    """
    
    def __init__(self, scales: Optional[np.ndarray] = None):
        self.scales = scales if scales is not None else np.logspace(0.1, 2, 20)

    def compute_susceptibility_spectrum(self, data: np.ndarray) -> Dict[float, float]:
        """
        Computes susceptibility chi at various scales using Continuous Wavelet Transform (CWT).
        
        Args:
            data: 1D array of the observable
            
        Returns:
            Dictionary mapping scale -> max_susceptibility
        """
        # Use Ricker wavelet (Mexican Hat) which approximates 2nd derivative
        # Susceptibility is related to fluctuations, which CWT captures
        widths = self.scales
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        
        spectrum = {}
        for i, scale in enumerate(widths):
            # The energy at this scale represents the magnitude of fluctuations
            # chi(scale) ~ <|W(s, t)|^2>
            chi_scale = np.mean(np.abs(cwtmatr[i, :])**2)
            spectrum[scale] = chi_scale
            
        return spectrum

    def find_critical_scales(self, spectrum: Dict[float, float]) -> List[float]:
        """
        Identifies scales with peak susceptibility.
        """
        scales = np.array(list(spectrum.keys()))
        chis = np.array(list(spectrum.values()))
        
        # Find peaks in the spectrum
        peaks, _ = signal.find_peaks(chis)
        return scales[peaks].tolist()
