#!/usr/bin/env python3
"""
Sigma-C Protein Stability Adapter
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Adapter for Protein Stability Analysis using the contraction
index sigma = D * gamma.

Implements the sigma framework for amyloid disease proteins,
validating the contraction-index theory against experimental
thermodynamic and mutational data (Paper 5).

For commercial licensing without AGPL-3.0 obligations, contact:
nfo@forgottenforge.xyz

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple


class DualBasinModel:
    """
    Simplified dual-basin Go model for protein folding.
    Two competing attractors: native fold and amyloid state.
    Energy: E(alpha) = (1-alpha)*E_nat + alpha*E_amy

    Args:
        N: Number of residues
        S: States per residue (default 8)
        contacts: Number of native contacts (default 12)
        epsilon_nat: Native contact energy (default 1.0)
        epsilon_amy: Amyloid contact energy (default 0.5)
    """

    def __init__(self, N: int = 30, S: int = 8, contacts: int = 12,
                 epsilon_nat: float = 1.0, epsilon_amy: float = 0.5):
        self.N = N
        self.S = S
        self.contacts = contacts
        self.epsilon_nat = epsilon_nat
        self.epsilon_amy = epsilon_amy
        # Generate random contact maps
        np.random.seed(42)
        all_pairs = [(i, j) for i in range(N) for j in range(i+2, N)]
        selected = np.random.choice(len(all_pairs), size=min(contacts, len(all_pairs)), replace=False)
        self.native_contacts = [all_pairs[s] for s in selected]
        self.amyloid_contacts = [all_pairs[s] for s in np.random.choice(len(all_pairs), size=min(contacts, len(all_pairs)), replace=False)]
        self.native_state = np.random.randint(0, S, size=N)
        self.amyloid_state = np.random.randint(0, S, size=N)

    def energy(self, config: np.ndarray, alpha: float) -> float:
        """Compute energy E(alpha) = (1-alpha)*E_nat + alpha*E_amy."""
        e_nat = -self.epsilon_nat * sum(
            1 for i, j in self.native_contacts
            if config[i] == self.native_state[i] and config[j] == self.native_state[j]
        )
        e_amy = -self.epsilon_amy * sum(
            1 for i, j in self.amyloid_contacts
            if config[i] == self.amyloid_state[i] and config[j] == self.amyloid_state[j]
        )
        return (1 - alpha) * e_nat + alpha * e_amy

    def native_contacts_fraction(self, config: np.ndarray) -> float:
        """Q_nat: fraction of native contacts formed."""
        if len(self.native_contacts) == 0:
            return 0.0
        formed = sum(
            1 for i, j in self.native_contacts
            if config[i] == self.native_state[i] and config[j] == self.native_state[j]
        )
        return formed / len(self.native_contacts)

    def simulate(self, alpha: float, n_steps: int = 3000, n_trials: int = 10,
                 temperature: float = 1.0) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation at given alpha.

        Returns:
            Dict with 'D', 'gamma', 'sigma', 'Q_nat_mean', 'Q_amy_mean'
        """
        Q_nats = []
        D_values = []
        gamma_values = []

        for trial in range(n_trials):
            config = np.random.randint(0, self.S, size=self.N)
            distances = []
            productive_moves = 0
            total_moves = 0

            for step in range(n_steps):
                # Propose single-residue flip
                site = np.random.randint(self.N)
                new_state = np.random.randint(self.S)
                old_state = config[site]

                e_old = self.energy(config, alpha)
                config[site] = new_state
                e_new = self.energy(config, alpha)

                # Metropolis acceptance
                delta_e = e_new - e_old
                if delta_e <= 0 or np.random.random() < np.exp(-delta_e / temperature):
                    # Accepted - check if productive (reduces distance to native)
                    d_new = 1.0 - self.native_contacts_fraction(config)
                    if len(distances) > 0 and d_new < distances[-1]:
                        productive_moves += 1
                    total_moves += 1
                    distances.append(d_new)
                else:
                    config[site] = old_state  # Reject
                    if len(distances) > 0:
                        distances.append(distances[-1])
                    total_moves += 1

            Q_nats.append(self.native_contacts_fraction(config))

            # D: fraction of productive moves
            D = productive_moves / max(total_moves, 1)
            D_values.append(D)

            # gamma: average contraction ratio
            if len(distances) >= 2:
                ratios = [distances[i+1] / (distances[i] + 1e-10) for i in range(len(distances)-1) if distances[i] > 0.01]
                gamma = np.mean(ratios) if ratios else 1.0
            else:
                gamma = 1.0
            gamma_values.append(gamma)

        D_mean = float(np.mean(D_values))
        gamma_mean = float(np.mean(gamma_values))
        sigma = D_mean * gamma_mean

        return {
            'D': D_mean,
            'gamma': gamma_mean,
            'sigma': sigma,
            'Q_nat_mean': float(np.mean(Q_nats)),
            'n_trials': n_trials,
            'n_steps': n_steps,
            'alpha': alpha,
        }

    def sweep_alpha(self, alpha_range: np.ndarray = None, **kwargs) -> List[Dict]:
        """Sweep alpha and compute sigma at each point."""
        if alpha_range is None:
            alpha_range = np.linspace(0, 1, 11)
        results = []
        for alpha in alpha_range:
            r = self.simulate(float(alpha), **kwargs)
            results.append(r)
        return results

    def find_critical_alpha(self, **kwargs) -> float:
        """Find alpha where sigma crosses 1.0."""
        results = self.sweep_alpha(np.linspace(0, 1, 21), **kwargs)
        for i in range(len(results) - 1):
            if results[i]['sigma'] < 1.0 and results[i+1]['sigma'] >= 1.0:
                # Linear interpolation
                s1, s2 = results[i]['sigma'], results[i+1]['sigma']
                a1, a2 = results[i]['alpha'], results[i+1]['alpha']
                return a1 + (1.0 - s1) / (s2 - s1) * (a2 - a1)
        return float('nan')


class ProteinAdapter(SigmaCAdapter):
    """
    Adapter for Protein Stability Analysis.

    Uses the contraction index sigma = D * gamma to quantify
    protein stability and predict amyloid disease onset.

    Units: kcal/mol throughout; R = 1.987e-3 kcal/(mol*K).
    """

    # Gas constant in kcal/(mol*K)
    R = 1.987e-3

    # ================================================================
    # Validation Data: TTR (Transthyretin) -- 25 mutations
    # ================================================================

    TTR_MUTATIONS = [
        {'name': 'T119M',  'ddG': -0.8, 'sigma': 0.917, 'onset': None, 'phenotype': 'protective'},
        {'name': 'G6S',    'ddG':  0.3, 'sigma': 0.930, 'onset': 72,   'phenotype': 'FAC'},
        {'name': 'V14A',   'ddG':  0.4, 'sigma': 0.931, 'onset': 70,   'phenotype': 'mixed'},
        {'name': 'V30M',   'ddG':  1.2, 'sigma': 0.941, 'onset': 33,   'phenotype': 'FAP-I'},
        {'name': 'V30A',   'ddG':  1.0, 'sigma': 0.938, 'onset': 40,   'phenotype': 'FAP'},
        {'name': 'V30G',   'ddG':  1.5, 'sigma': 0.944, 'onset': 30,   'phenotype': 'FAP'},
        {'name': 'V30L',   'ddG':  0.8, 'sigma': 0.936, 'onset': 50,   'phenotype': 'FAP'},
        {'name': 'T49A',   'ddG':  0.5, 'sigma': 0.932, 'onset': 65,   'phenotype': 'mixed'},
        {'name': 'S50R',   'ddG':  0.6, 'sigma': 0.933, 'onset': 60,   'phenotype': 'FAP'},
        {'name': 'S52P',   'ddG':  1.8, 'sigma': 0.947, 'onset': 28,   'phenotype': 'FAP'},
        {'name': 'E54G',   'ddG':  0.7, 'sigma': 0.935, 'onset': 55,   'phenotype': 'mixed'},
        {'name': 'L55P',   'ddG':  2.5, 'sigma': 0.955, 'onset': 20,   'phenotype': 'FAP-aggressive'},
        {'name': 'L58H',   'ddG':  1.6, 'sigma': 0.945, 'onset': 32,   'phenotype': 'FAP'},
        {'name': 'T60A',   'ddG':  1.3, 'sigma': 0.942, 'onset': 45,   'phenotype': 'FAC'},
        {'name': 'E89Q',   'ddG':  0.4, 'sigma': 0.931, 'onset': 68,   'phenotype': 'FAC'},
        {'name': 'A97S',   'ddG':  0.9, 'sigma': 0.937, 'onset': 48,   'phenotype': 'mixed'},
        {'name': 'R104H',  'ddG':  0.5, 'sigma': 0.932, 'onset': 62,   'phenotype': 'FAC'},
        {'name': 'I107V',  'ddG':  1.1, 'sigma': 0.940, 'onset': 35,   'phenotype': 'FAP'},
        {'name': 'A109T',  'ddG':  0.3, 'sigma': 0.930, 'onset': 75,   'phenotype': 'FAC'},
        {'name': 'S112I',  'ddG':  0.6, 'sigma': 0.933, 'onset': 58,   'phenotype': 'FAC'},
        {'name': 'Y114C',  'ddG':  1.4, 'sigma': 0.943, 'onset': 38,   'phenotype': 'FAP'},
        {'name': 'V122I',  'ddG':  0.5, 'sigma': 0.932, 'onset': 65,   'phenotype': 'FAC'},
        {'name': 'V122A',  'ddG':  0.7, 'sigma': 0.935, 'onset': 55,   'phenotype': 'FAC'},
        {'name': 'D18G',   'ddG':  1.7, 'sigma': 0.946, 'onset': 25,   'phenotype': 'leptomeningeal'},
        {'name': 'A25T',   'ddG':  1.0, 'sigma': 0.938, 'onset': 42,   'phenotype': 'CNS'},
    ]

    TTR_PARAMS = {'N': 127, 'T': 310, 'delta_G_wt': 25.0}  # kcal/mol -> use 6.0 kcal/mol

    # ================================================================
    # Validation Data: Lysozyme (LYZ) -- 6 mutations
    # ================================================================

    LYZ_MUTATIONS = [
        {'name': 'I56T',   'ddG': 2.1, 'sigma': 0.950, 'onset': 50, 'phenotype': 'systemic_amyloidosis'},
        {'name': 'D67H',   'ddG': 1.8, 'sigma': 0.947, 'onset': 55, 'phenotype': 'systemic_amyloidosis'},
        {'name': 'F57I',   'ddG': 1.5, 'sigma': 0.944, 'onset': 48, 'phenotype': 'systemic_amyloidosis'},
        {'name': 'W64R',   'ddG': 2.5, 'sigma': 0.955, 'onset': 40, 'phenotype': 'systemic_amyloidosis'},
        {'name': 'T70N',   'ddG': 1.2, 'sigma': 0.941, 'onset': 60, 'phenotype': 'renal_amyloidosis'},
        {'name': 'F57I/T70N', 'ddG': 2.8, 'sigma': 0.958, 'onset': 35, 'phenotype': 'aggressive'},
    ]

    LYZ_PARAMS = {'N': 130, 'delta_G_wt': 9.3}  # 9.3 kcal/mol (38.9 kJ/mol)

    # ================================================================
    # Validation Data: Gelsolin (GSN) -- 5 mutations
    # ================================================================

    GSN_MUTATIONS = [
        {'name': 'D187N',  'ddG': 1.5, 'sigma': 0.944, 'onset': 40, 'phenotype': 'Finnish_amyloidosis'},
        {'name': 'D187Y',  'ddG': 1.8, 'sigma': 0.947, 'onset': 35, 'phenotype': 'Finnish_amyloidosis'},
        {'name': 'G167R',  'ddG': 1.0, 'sigma': 0.938, 'onset': 50, 'phenotype': 'lattice_dystrophy'},
        {'name': 'N184K',  'ddG': 0.8, 'sigma': 0.936, 'onset': 55, 'phenotype': 'mild'},
        {'name': 'A174S',  'ddG': 0.5, 'sigma': 0.932, 'onset': 65, 'phenotype': 'subclinical'},
    ]

    GSN_PARAMS = {'N': 117, 'delta_G_wt': 6.0}  # kcal/mol (25.1 kJ/mol)

    # ================================================================
    # Negative Control: SOD1 (Superoxide Dismutase 1) -- 10 mutations
    # ================================================================

    SOD1_MUTATIONS = [
        {'name': 'A4V',    'ddG': 3.5, 'sigma': 0.965, 'onset': 47, 'phenotype': 'ALS_aggressive'},
        {'name': 'G93A',   'ddG': 1.0, 'sigma': 0.938, 'onset': 45, 'phenotype': 'ALS'},
        {'name': 'D90A',   'ddG': 0.5, 'sigma': 0.932, 'onset': 44, 'phenotype': 'ALS_slow'},
        {'name': 'H46R',   'ddG': 2.0, 'sigma': 0.949, 'onset': 55, 'phenotype': 'ALS_slow'},
        {'name': 'G85R',   'ddG': 2.8, 'sigma': 0.958, 'onset': 40, 'phenotype': 'ALS'},
        {'name': 'A89V',   'ddG': 1.5, 'sigma': 0.944, 'onset': 52, 'phenotype': 'ALS'},
        {'name': 'E100G',  'ddG': 1.2, 'sigma': 0.941, 'onset': 48, 'phenotype': 'ALS'},
        {'name': 'I113T',  'ddG': 0.8, 'sigma': 0.936, 'onset': 58, 'phenotype': 'ALS_variable'},
        {'name': 'L38V',   'ddG': 1.8, 'sigma': 0.947, 'onset': 43, 'phenotype': 'ALS'},
        {'name': 'G37R',   'ddG': 1.3, 'sigma': 0.942, 'onset': 35, 'phenotype': 'ALS_aggressive'},
    ]

    SOD1_PARAMS = {'N': 153, 'delta_G_wt': 13.5}  # kcal/mol (56.5 kJ/mol)

    # ================================================================
    # Negative Control: Prion Protein (PRNP) -- 7 mutations
    # ================================================================

    PRNP_MUTATIONS = [
        {'name': 'E200K',  'ddG': 0.8, 'sigma': 0.936, 'onset': 58, 'phenotype': 'CJD'},
        {'name': 'D178N',  'ddG': 1.5, 'sigma': 0.944, 'onset': 50, 'phenotype': 'FFI'},
        {'name': 'P102L',  'ddG': 0.5, 'sigma': 0.932, 'onset': 45, 'phenotype': 'GSS'},
        {'name': 'V210I',  'ddG': 0.3, 'sigma': 0.930, 'onset': 55, 'phenotype': 'CJD'},
        {'name': 'T183A',  'ddG': 1.0, 'sigma': 0.938, 'onset': 42, 'phenotype': 'CJD'},
        {'name': 'F198S',  'ddG': 1.2, 'sigma': 0.941, 'onset': 52, 'phenotype': 'GSS'},
        {'name': 'Q217R',  'ddG': 0.6, 'sigma': 0.933, 'onset': 62, 'phenotype': 'GSS'},
    ]

    PRNP_PARAMS = {'N': 104, 'delta_G_wt': 6.0}  # kcal/mol (25.1 kJ/mol)

    # ================================================================
    # Reference Proteins: Small globular proteins
    # ================================================================

    REFERENCE_PROTEINS = [
        {'name': 'Trp-cage',   'N': 20,  'T_m': 317, 'sigma_310': 0.933},
        {'name': 'Villin HP35', 'N': 35, 'T_m': 342, 'sigma_310': 0.898},
        {'name': 'CI2',        'N': 64,  'T_m': 337, 'sigma_310': 0.904},
        {'name': 'ACBP',       'N': 86,  'T_m': 332, 'sigma_310': 0.914},
    ]

    CROSS_PROTEIN_VALIDATION = [
        # Barnase (N=110)
        {'name': 'A32G', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 4.6, 'ddG': 1.10, 'destabilizing': True},
        {'name': 'I51A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 13.4, 'ddG': 3.20, 'destabilizing': True},
        {'name': 'I76A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 12.1, 'ddG': 2.89, 'destabilizing': True},
        {'name': 'I88A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 8.8, 'ddG': 2.10, 'destabilizing': True},
        {'name': 'V10A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 7.1, 'ddG': 1.70, 'destabilizing': True},
        {'name': 'L14A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': 10.9, 'ddG': 2.60, 'destabilizing': True},
        {'name': 'N58A', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': -0.4, 'ddG': -0.10, 'destabilizing': False},
        {'name': 'D93N', 'protein': 'Barnase', 'N': 110, 'ddG_kJmol': -1.7, 'ddG': -0.41, 'destabilizing': False},
        # CI2 (N=64)
        {'name': 'I20A', 'protein': 'CI2', 'N': 64, 'ddG_kJmol': 10.0, 'ddG': 2.39, 'destabilizing': True},
        {'name': 'L32A', 'protein': 'CI2', 'N': 64, 'ddG_kJmol': 12.6, 'ddG': 3.01, 'destabilizing': True},
        {'name': 'V47A', 'protein': 'CI2', 'N': 64, 'ddG_kJmol': 3.3, 'ddG': 0.79, 'destabilizing': True},
        {'name': 'D52A', 'protein': 'CI2', 'N': 64, 'ddG_kJmol': -0.8, 'ddG': -0.19, 'destabilizing': False},
        # T4 Lysozyme (N=164)
        {'name': 'L99A', 'protein': 'T4Lys', 'N': 164, 'ddG_kJmol': 22.2, 'ddG': 5.31, 'destabilizing': True},
        {'name': 'L99G', 'protein': 'T4Lys', 'N': 164, 'ddG_kJmol': 27.2, 'ddG': 6.50, 'destabilizing': True},
        {'name': 'A98V', 'protein': 'T4Lys', 'N': 164, 'ddG_kJmol': -2.5, 'ddG': -0.60, 'destabilizing': False},
        {'name': 'V149I', 'protein': 'T4Lys', 'N': 164, 'ddG_kJmol': -1.3, 'ddG': -0.31, 'destabilizing': False},
        # Human Lysozyme (N=130)
        {'name': 'I56T_xval', 'protein': 'HumanLyz', 'N': 130, 'ddG_kJmol': 25.1, 'ddG': 6.00, 'destabilizing': True},
        {'name': 'D67H_xval', 'protein': 'HumanLyz', 'N': 130, 'ddG_kJmol': 18.8, 'ddG': 4.49, 'destabilizing': True},
        {'name': 'W64R_xval', 'protein': 'HumanLyz', 'N': 130, 'ddG_kJmol': 12.6, 'ddG': 3.01, 'destabilizing': True},
        {'name': 'T70N_xval', 'protein': 'HumanLyz', 'N': 130, 'ddG_kJmol': 5.0, 'ddG': 1.20, 'destabilizing': True},
    ]

    APP_VUS_PREDICTIONS = [
        {'name': 'G696R', 'ddG': 2.87, 'risk': 'high'},
        {'name': 'D678H', 'ddG': 2.33, 'risk': 'high'},
        {'name': 'G700E', 'ddG': 2.21, 'risk': 'high'},
        {'name': 'K687E', 'ddG': 2.19, 'risk': 'high'},
        {'name': 'G700D', 'ddG': 1.94, 'risk': 'high'},
        {'name': 'V695A', 'ddG': 1.73, 'risk': 'high'},
        {'name': 'A692T', 'ddG': 1.52, 'risk': 'high'},
        {'name': 'G696S', 'ddG': 0.91, 'risk': 'moderate'},
        {'name': 'G709S', 'ddG': 0.91, 'risk': 'moderate'},
        {'name': 'V711L', 'ddG': 0.89, 'risk': 'moderate'},
        {'name': 'K687R', 'ddG': 0.73, 'risk': 'moderate'},
        {'name': 'V695I', 'ddG': 0.68, 'risk': 'moderate'},
        {'name': 'E682G', 'ddG': 0.35, 'risk': 'low'},
        {'name': 'A692V', 'ddG': 0.29, 'risk': 'low'},
        {'name': 'G681A', 'ddG': 0.13, 'risk': 'low'},
        {'name': 'G696V', 'ddG': -0.08, 'risk': 'stabilizing'},
    ]

    # ================================================================

    def __init__(self, protein_name: Optional[str] = None,
                 N: Optional[int] = None,
                 T: float = 310.0,
                 delta_G_wt: Optional[float] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ProteinAdapter.

        Args:
            protein_name: Name of the protein system (e.g. 'TTR', 'LYZ').
            N: Number of residues in the protein.
            T: Temperature in Kelvin (default 310 K, body temperature).
            delta_G_wt: Wild-type stability in kcal/mol.
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.protein_name = protein_name
        self.N = N
        self.T = T
        self.delta_G_wt = delta_G_wt

        # Auto-load parameters from built-in datasets
        if protein_name is not None:
            params_key = f"{protein_name.upper()}_PARAMS"
            if hasattr(self, params_key):
                params = getattr(self, params_key)
                if self.N is None:
                    self.N = params['N']
                if self.delta_G_wt is None:
                    self.delta_G_wt = params.get('delta_G_wt')
                if 'T' in params and T == 310.0:
                    self.T = params['T']

    # ================================================================
    # Core Methods
    # ================================================================

    def get_observable(self, data: Any, **kwargs) -> float:
        """
        Calculate the protein stability observable sigma.

        If data is a dict with 'delta_delta_G' and 'N', returns the
        mutational contraction index sigma_mutation.

        If data is a float, interprets it as delta_G and returns the
        thermodynamic contraction index sigma_thermodynamic.

        Args:
            data: Either a dict {'delta_delta_G': float, 'N': int}
                  or a float (delta_G value in kcal/mol).

        Returns:
            The contraction index sigma.
        """
        if isinstance(data, dict):
            ddG = data['delta_delta_G']
            N = data.get('N', self.N)
            if N is None:
                raise ValueError("N (number of residues) must be provided")
            return self.sigma_mutation(ddG, N)
        elif isinstance(data, (int, float)):
            N = kwargs.get('N', self.N)
            if N is None:
                raise ValueError("N (number of residues) must be provided")
            return self.sigma_thermodynamic(float(data), N)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def sigma_product(self, D: float, gamma: float) -> float:
        """
        Compute the contraction index as sigma = D * gamma.

        Args:
            D: Diffusion-like parameter.
            gamma: Damping-like parameter.

        Returns:
            The contraction index sigma.
        """
        return D * gamma

    def sigma_thermodynamic(self, delta_G: float, N: int,
                            T: Optional[float] = None) -> float:
        """
        Compute the thermodynamic contraction index.

        sigma = exp(-delta_G / (N * R * T))

        Args:
            delta_G: Free energy of unfolding in kcal/mol.
            N: Number of residues.
            T: Temperature in Kelvin (default: self.T).

        Returns:
            Thermodynamic contraction index.
        """
        T = T or self.T
        return float(np.exp(-delta_G / (N * self.R * T)))

    def sigma_mutation(self, delta_delta_G: float, N: int,
                       T: Optional[float] = None) -> float:
        """
        Compute the mutational contraction index.

        sigma = exp(delta_delta_G / (N * R * T))

        A destabilizing mutation (positive ddG) yields sigma > 1,
        indicating enhanced aggregation propensity.

        Args:
            delta_delta_G: Change in free energy due to mutation (kcal/mol).
            N: Number of residues.
            T: Temperature in Kelvin (default: self.T).

        Returns:
            Mutational contraction index.
        """
        T = T or self.T
        return float(np.exp(delta_delta_G / (N * self.R * T)))

    def sigma_drift(self, sigma_baseline: float, age: float,
                    rate: float = 0.003) -> float:
        """
        Compute age-dependent sigma drift.

        Models the accumulation of proteostatic stress over time,
        with a reference age of 30 years.

        sigma(age) = sigma_baseline + rate * (age - 30) / 10

        Args:
            sigma_baseline: Baseline contraction index at age 30.
            age: Current age in years.
            rate: Drift rate per decade (default 0.003).

        Returns:
            Age-adjusted contraction index.
        """
        return sigma_baseline + rate * (age - 30) / 10.0

    def predict_onset(self, sigma_baseline: float,
                      rate: float = 0.003) -> float:
        """
        Predict disease onset age from sigma drift model.

        Finds the age at which sigma_drift reaches the critical
        threshold of 1.0.

        Args:
            sigma_baseline: Baseline contraction index at age 30.
            rate: Drift rate per decade (default 0.003).

        Returns:
            Predicted onset age in years.
        """
        if sigma_baseline >= 1.0:
            return 30.0
        decades_to_threshold = (1.0 - sigma_baseline) / rate
        return 30.0 + decades_to_threshold * 10.0

    def onset_envelope(self, sigma_baseline: float,
                       rate_range: Tuple[float, float] = (0.02, 0.05)) -> Tuple[float, float]:
        """
        Compute an onset envelope (earliest, latest) for a range
        of drift rates.

        Args:
            sigma_baseline: Baseline contraction index.
            rate_range: Tuple of (low_rate, high_rate) per decade.

        Returns:
            Tuple of (earliest_onset, latest_onset) in years.
        """
        low = self.predict_onset(sigma_baseline, rate_range[1])
        high = self.predict_onset(sigma_baseline, rate_range[0])
        return (low, high)

    def classify_mechanism(self, protein_data: dict) -> dict:
        """
        Classify the disease mechanism for a protein system.

        Categories:
        - 'IDP': Intrinsically disordered protein (no stable fold).
        - 'stability_driven': Destabilizing mutations drive aggregation.
        - 'gain_of_function': Mutations confer toxic function.
        - 'templated_conversion': Prion-like templated misfolding.

        Args:
            protein_data: Dict with keys such as 'has_stable_fold',
                          'delta_G', 'mutations_destabilizing',
                          'gain_of_function', 'templated'.

        Returns:
            Dict with 'mechanism', 'sigma_applicable', and 'rationale'.
        """
        has_fold = protein_data.get('has_stable_fold', True)
        delta_G = protein_data.get('delta_G', 0.0)
        destabilizing = protein_data.get('mutations_destabilizing', False)
        gof = protein_data.get('gain_of_function', False)
        templated = protein_data.get('templated', False)

        if not has_fold:
            return {
                'mechanism': 'IDP',
                'sigma_applicable': False,
                'rationale': ('Intrinsically disordered proteins lack a '
                              'stable native fold; sigma analysis requires '
                              'a defined folded state.')
            }

        if templated:
            return {
                'mechanism': 'templated_conversion',
                'sigma_applicable': False,
                'rationale': ('Templated conversion (prion-like) mechanisms '
                              'are driven by inter-molecular contacts, not '
                              'single-molecule thermodynamic instability.')
            }

        if gof:
            return {
                'mechanism': 'gain_of_function',
                'sigma_applicable': False,
                'rationale': ('Gain-of-function mutations confer toxic '
                              'properties independent of thermodynamic '
                              'destabilization; sigma analysis is not the '
                              'primary predictor.')
            }

        if delta_G > 0 or destabilizing:
            return {
                'mechanism': 'stability_driven',
                'sigma_applicable': True,
                'rationale': ('Destabilizing mutations reduce native-state '
                              'stability, populating partially unfolded '
                              'intermediates that aggregate. Sigma analysis '
                              'is directly applicable.')
            }

        return {
            'mechanism': 'unknown',
            'sigma_applicable': True,
            'rationale': ('Mechanism not definitively classified; sigma '
                          'analysis may still provide useful insight.')
        }

    def validate_scope(self, protein_data: dict) -> dict:
        """
        Check whether sigma analysis is applicable to the given
        protein system.

        Args:
            protein_data: Dict with protein characteristics.

        Returns:
            Dict with 'in_scope', 'warnings', and 'details'.
        """
        warnings_list: List[str] = []
        details: Dict[str, Any] = {}

        N = protein_data.get('N', self.N)
        has_fold = protein_data.get('has_stable_fold', True)
        mechanism = protein_data.get('mechanism', None)

        in_scope = True

        if not has_fold:
            in_scope = False
            warnings_list.append('Protein is intrinsically disordered; '
                                 'sigma analysis requires a stable fold.')

        if N is not None:
            details['N'] = N
            if N < 15:
                warnings_list.append(f'Very short chain (N={N}); sigma '
                                     'analysis may not be reliable.')
            if N > 1000:
                warnings_list.append(f'Very large protein (N={N}); '
                                     'multi-domain effects may dominate.')

        if mechanism in ('gain_of_function', 'templated_conversion'):
            in_scope = False
            warnings_list.append(f'Mechanism "{mechanism}" is outside '
                                 'the scope of sigma stability analysis.')

        return {
            'in_scope': in_scope,
            'warnings': warnings_list,
            'details': details,
        }

    def analyze_protein(self, mutations: List[Dict[str, Any]]) -> dict:
        """
        Comprehensive analysis of a set of mutations.

        For each mutation dict with keys 'name', 'delta_delta_G', and
        optionally 'onset_observed', computes sigma, drift predictions,
        and correlation statistics.

        Args:
            mutations: List of mutation dicts. Each must have:
                - 'name': str
                - 'delta_delta_G': float (kcal/mol)
                Optional:
                - 'onset_observed': float (years)
                - 'N': int (overrides self.N)

        Returns:
            Dict with 'mutations', 'correlation', 'summary'.
        """
        N = self.N
        if N is None:
            raise ValueError("N (number of residues) must be set on the "
                             "adapter or provided per mutation.")

        results = []
        sigmas = []
        onsets_observed = []
        onsets_predicted = []

        for mut in mutations:
            name = mut['name']
            ddG = mut.get('delta_delta_G', mut.get('ddG'))
            if ddG is None:
                raise KeyError(f"Mutation {name} missing 'ddG' or 'delta_delta_G'")
            n = mut.get('N', N)
            onset_obs = mut.get('onset_observed', mut.get('onset'))

            sigma = self.sigma_mutation(ddG, n)
            onset_pred_point = self.predict_onset(sigma)
            envelope = self.onset_envelope(sigma)

            entry = {
                'name': name,
                'delta_delta_G': ddG,
                'N': n,
                'sigma': sigma,
                'onset_predicted': onset_pred_point,
                'onset_envelope': envelope,
            }

            if onset_obs is not None:
                entry['onset_observed'] = onset_obs
                sigmas.append(sigma)
                onsets_observed.append(onset_obs)
                onsets_predicted.append(onset_pred_point)

            results.append(entry)

        # Compute Spearman correlation between sigma and observed onset
        correlation = {}
        if len(sigmas) >= 3:
            rho_sigma_onset, p_sigma_onset = stats.spearmanr(
                sigmas, onsets_observed
            )
            correlation['sigma_vs_onset'] = {
                'spearman_rho': float(rho_sigma_onset),
                'p_value': float(p_sigma_onset),
                'n': len(sigmas),
            }

            if len(onsets_predicted) >= 3 and len(set(onsets_predicted)) > 1:
                rho_pred, p_pred = stats.spearmanr(
                    onsets_predicted, onsets_observed
                )
                correlation['predicted_vs_observed_onset'] = {
                    'spearman_rho': float(rho_pred),
                    'p_value': float(p_pred),
                    'n': len(onsets_predicted),
                }

        # Summary statistics
        all_sigmas = [r['sigma'] for r in results]
        summary = {
            'n_mutations': len(results),
            'sigma_mean': float(np.mean(all_sigmas)),
            'sigma_std': float(np.std(all_sigmas)),
            'sigma_min': float(np.min(all_sigmas)),
            'sigma_max': float(np.max(all_sigmas)),
        }

        return {
            'protein': self.protein_name,
            'N': N,
            'T': self.T,
            'R': self.R,
            'mutations': results,
            'correlation': correlation,
            'summary': summary,
        }

    # ================================================================
    # v1.1.0: Domain-Specific Hooks
    # ================================================================

    def _domain_specific_diagnose(self, data: Any = None,
                                  **kwargs) -> Dict[str, Any]:
        """Protein-specific diagnostics."""
        issues: List[str] = []
        recommendations: List[str] = []
        details: Dict[str, Any] = {}

        if self.N is None:
            issues.append('N (number of residues) is not set.')
            recommendations.append('Set N via constructor or provide '
                                   'protein_name to auto-load parameters.')

        if self.delta_G_wt is None:
            issues.append('Wild-type stability (delta_G_wt) is not set.')
            recommendations.append('Set delta_G_wt via constructor or '
                                   'provide protein_name.')

        if data is not None:
            if isinstance(data, list):
                details['n_mutations'] = len(data)
                if len(data) < 3:
                    issues.append(f'Only {len(data)} mutations provided; '
                                  'need >= 3 for correlation analysis.')
                    recommendations.append('Provide at least 3 mutations '
                                           'with onset data for meaningful '
                                           'statistics.')
                # Check for required keys
                for i, mut in enumerate(data):
                    if 'delta_delta_G' not in mut:
                        issues.append(f'Mutation {i} missing '
                                      '"delta_delta_G" key.')
            elif isinstance(data, dict):
                if 'delta_delta_G' not in data and 'ddG' not in data:
                    issues.append('Data dict missing "delta_delta_G" key.')

        details['protein_name'] = self.protein_name
        details['N'] = self.N
        details['T'] = self.T

        status = 'ok'
        if issues:
            status = 'warning'
            if any('not set' in i for i in issues):
                status = 'error'

        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'auto_fix': None,
            'details': details,
        }

    def _domain_specific_validate(self, data: Any = None,
                                  **kwargs) -> Dict[str, bool]:
        """Validate protein analysis techniques."""
        checks: Dict[str, bool] = {}

        checks['N_set'] = self.N is not None
        checks['T_positive'] = self.T > 0
        checks['R_correct'] = abs(self.R - 1.987e-3) < 1e-8

        if data is not None and isinstance(data, list):
            checks['has_mutations'] = len(data) > 0
            checks['sufficient_for_correlation'] = len(data) >= 3

            has_onset = sum(1 for m in data
                           if m.get('onset_observed') is not None
                           or m.get('onset') is not None)
            checks['has_onset_data'] = has_onset > 0

            all_have_ddG = all('delta_delta_G' in m or 'ddG' in m
                               for m in data)
            checks['all_have_ddG'] = all_have_ddG

        return checks

    def _domain_specific_explain(self, result: Dict[str, Any],
                                 **kwargs) -> str:
        """Protein-specific explanation of results."""
        protein = result.get('protein', 'Unknown')
        N = result.get('N', 'N/A')
        T = result.get('T', 'N/A')
        summary = result.get('summary', {})
        correlation = result.get('correlation', {})

        lines = [
            '# Protein Stability Analysis Results',
            '',
            f'**Protein:** {protein}',
            f'**Residues (N):** {N}',
            f'**Temperature:** {T} K',
            '',
            '## Summary',
            f'- Mutations analyzed: {summary.get("n_mutations", "N/A")}',
            f'- Sigma mean: {summary.get("sigma_mean", "N/A"):.4f}'
            if isinstance(summary.get('sigma_mean'), float) else
            f'- Sigma mean: N/A',
            f'- Sigma range: [{summary.get("sigma_min", "N/A"):.4f}, '
            f'{summary.get("sigma_max", "N/A"):.4f}]'
            if isinstance(summary.get('sigma_min'), float) else
            f'- Sigma range: N/A',
            '',
        ]

        if 'sigma_vs_onset' in correlation:
            corr = correlation['sigma_vs_onset']
            lines.extend([
                '## Correlation (sigma vs onset)',
                f'- Spearman rho: {corr["spearman_rho"]:.3f}',
                f'- p-value: {corr["p_value"]:.2e}',
                f'- n: {corr["n"]}',
                '',
            ])

        lines.extend([
            '## Interpretation',
            '- sigma < 1.0: native state is thermodynamically favored',
            '- sigma ~ 1.0: marginal stability, onset imminent',
            '- sigma > 1.0: native state is destabilized, aggregation '
            'expected',
            '- Higher sigma correlates with earlier disease onset',
            '',
            'Negative Spearman rho (sigma vs onset) confirms that '
            'higher sigma predicts earlier onset.',
        ])

        return '\n'.join(lines)
