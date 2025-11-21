#!/usr/bin/env python3
"""
Sigma-C Seismic Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Seismic Catalogs and Criticality Analysis.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, Optional, List
from pathlib import Path
import requests
import hashlib
import time

class SeismicAdapter(SigmaCAdapter):
    """
    Adapter for Seismic Data (USGS).
    Ported from sigma_cize.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_dir = Path(self.config.get('cache_dir', 'cache_seis'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_catalog(self, region='california', years=10, min_mag=2.5) -> pd.DataFrame:
        """
        Fetch earthquake catalog from USGS.
        """
        regions = {
            'california': {'minlat': 32.5, 'maxlat': 42.0, 'minlon': -125.0, 'maxlon': -114.0},
            'japan': {'minlat': 30.0, 'maxlat': 46.0, 'minlon': 130.0, 'maxlon': 146.0},
            'chile': {'minlat': -45.0, 'maxlat': -17.0, 'minlon': -76.0, 'maxlon': -66.0},
        }
        bounds = regions.get(region, regions['california'])
        
        # Cache check
        cache_key = hashlib.md5(f"{region}_{years}_{min_mag}".encode()).hexdigest()
        cache_file = self.cache_dir / f"catalog_{cache_key}.csv"
        
        if cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=['time'])
            
        # Download logic (simplified for adapter)
        # In production, this should handle pagination/chunking like sigma_cize.py
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'csv',
            'starttime': (pd.Timestamp.now() - pd.Timedelta(days=365*years)).strftime('%Y-%m-%d'),
            'endtime': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'minmagnitude': min_mag,
            'minlatitude': bounds['minlat'],
            'maxlatitude': bounds['maxlat'],
            'minlongitude': bounds['minlon'],
            'maxlongitude': bounds['maxlon'],
            'orderby': 'time'
        }
        
        try:
            df = pd.read_csv(f"{url}?{'&'.join(f'{k}={v}' for k,v in params.items())}")
            df['time'] = pd.to_datetime(df['time'])
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            print(f"Error downloading catalog: {e}")
            return pd.DataFrame()

    def compute_stress_proxy(self, catalog: pd.DataFrame, resolution_km=10.0) -> np.ndarray:
        """
        Compute stress proxy field.
        """
        if len(catalog) == 0:
            return np.zeros((10, 10))
            
        lat_min, lat_max = catalog['latitude'].min(), catalog['latitude'].max()
        lon_min, lon_max = catalog['longitude'].min(), catalog['longitude'].max()
        
        km_per_deg = 111.0
        res_deg = resolution_km / km_per_deg
        
        n_lat = int((lat_max - lat_min) / res_deg) + 1
        n_lon = int((lon_max - lon_min) / res_deg) + 1
        
        lat_grid = np.linspace(lat_min, lat_max, n_lat)
        lon_grid = np.linspace(lon_min, lon_max, n_lon)
        
        stress = np.zeros((n_lat, n_lon))
        
        # Simplified accumulation
        for _, event in catalog.iterrows():
            energy = 10 ** (1.5 * event['mag'] + 4.8)
            lat_idx = np.abs(lat_grid - event['latitude']).argmin()
            lon_idx = np.abs(lon_grid - event['longitude']).argmin()
            if 0 <= lat_idx < n_lat and 0 <= lon_idx < n_lon:
                stress[lat_idx, lon_idx] += energy
                
        # Log transform and smooth
        stress = np.log10(stress + 1)
        stress = gaussian_filter(stress, sigma=1.5)
        return stress / (stress.max() + 1e-10)

    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Compute observable from stress field.
        """
        # data is the stress field
        # For seismic, observable is usually a property of the smoothed field
        # Here we implement the 'designed peak' logic from sigma_cize.py for robustness
        p90 = np.percentile(data.flatten(), 90)
        p10 = np.percentile(data.flatten(), 10)
        return float(p90 - p10)
        
    def analyze_criticality(self, catalog: pd.DataFrame, theory_mode='standard'):
        """
        Run criticality analysis.
        """
        stress_field = self.compute_stress_proxy(catalog)
        
        sigma_values = np.linspace(0.5, 5.0, 20)
        observables = []
        
        for sigma in sigma_values:
            smoothed = gaussian_filter(stress_field, sigma=sigma)
            obs = self.get_observable(smoothed)
            observables.append(obs)
            
        # Pre-processing for theory modes
        obs_array = np.array(observables)
        if theory_mode == 'temporal':
            obs_array = np.power(obs_array, 1.2)
        elif theory_mode == 'magnitude':
            obs_array = np.log1p(obs_array * 10) / np.log(11)
            
        analysis = self.compute_susceptibility(sigma_values, obs_array)
        
        return {
            'sigma_c': analysis['sigma_c'],
            'kappa': analysis['kappa'],
            'sigma_values': sigma_values,
            'observable': obs_array
        }
