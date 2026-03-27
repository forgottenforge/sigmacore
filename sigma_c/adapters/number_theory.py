#!/usr/bin/env python3
"""
Sigma-C Number Theory Adapter
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Adapter for Number Theory Analysis based on contraction geometry
of Collatz-type maps. Implements contraction defect, drift, and
countdown decomposition analysis for arbitrary qn+c maps.

For commercial licensing without AGPL-3.0 obligations, contact:
nfo@forgottenforge.xyz

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from ..core.base import SigmaCAdapter
from ..core.contraction import (
    v2,
    odd_part,
    single_step_map,
    cycle_map,
    embedding_depth,
    compute_contraction_defect,
    compute_drift,
    classify_map,
    sweep_modular_resolution,
    contraction_principle_check,
    countdown_decomposition,
    deterministic_trajectory,
    TWELVE_MAP_PREDICTIONS,
)
import numpy as np
from typing import Dict, Any, List, Optional
import random


class NumberTheoryAdapter(SigmaCAdapter):
    """
    Adapter for Number Theory analysis via contraction geometry.

    Analyzes Collatz-type maps n -> odd(q*n + c) using the
    contraction defect D_M and drift gamma_M framework to
    predict convergent vs divergent behavior.
    """

    # Reference data for validation (Collatz 3n+1, cycle map)
    D_M_REFERENCE = {4: 2.0, 6: 1.78, 8: 2.0, 10: 2.048, 12: 2.066, 14: 2.061, 16: 2.058}
    GAMMA_REFERENCE = {8: 0.5625, 10: 0.5625, 12: 0.5625, 14: 0.5625}

    def __init__(self, map_type: str = 'collatz', q: int = 3, c: int = 1,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the number theory adapter.

        Args:
            map_type: Type of map to analyze. One of:
                - 'collatz_single': single-step map n -> odd(q*n + c)
                - 'collatz_cycle': cycle map F(n) = odd(3^L * (n+1)/2^L - 1)
                - 'collatz': alias for 'collatz_cycle' (default)
                - 'custom': single-step map with custom q, c
            q: Multiplier for qn+c maps (default 3)
            c: Additive constant for qn+c maps (default 1)
            config: Optional configuration dictionary
        """
        super().__init__(config=config)
        self.map_type = map_type
        self.q = q
        self.c = c

        # Create the map function based on type
        if map_type == 'collatz_single':
            self.f = lambda n: single_step_map(n, q, c)
        elif map_type in ('collatz_cycle', 'collatz'):
            self.f = cycle_map
        elif map_type == 'custom':
            self.f = lambda n: single_step_map(n, q, c)
        else:
            self.f = lambda n: single_step_map(n, q, c)

    def get_observable(self, data, **kwargs) -> float:
        """
        Compute the observable from number-theoretic data.

        For an integer, returns embedding_depth(n) / 10.0.
        For a list/array of integers, returns mean embedding depth / 10.0.

        Args:
            data: An integer or list/array of integers (odd positive).

        Returns:
            Normalized embedding depth as a float.
        """
        if isinstance(data, (int, np.integer)):
            return embedding_depth(int(data)) / 10.0
        else:
            eds = [embedding_depth(int(n)) for n in data]
            return float(np.mean(eds)) / 10.0 if len(eds) > 0 else 0.0

    def compute_D_M(self, M: int) -> float:
        """
        Compute the contraction defect D_M = |S_M| / |f(S_M)| at resolution M.

        Args:
            M: Modular resolution (residue classes mod 2^M)

        Returns:
            D_M >= 1.0 (structurally non-injective if > 1)
        """
        return compute_contraction_defect(self.f, M)

    def compute_gamma_M(self, M: int) -> float:
        """
        Compute the drift gamma_M at resolution M.

        Args:
            M: Modular resolution

        Returns:
            gamma_M: average multiplicative growth per step
        """
        return compute_drift(self.f, M)

    def sweep_resolution(self, M_range: range = range(4, 17)) -> List[Dict]:
        """
        Compute (D_M, gamma_M) for a range of modular resolutions.

        Args:
            M_range: Range of M values to sweep (default 4..16)

        Returns:
            List of dicts with 'M', 'D_M', 'gamma_M', 'domain_size', 'image_size'
        """
        return sweep_modular_resolution(self.f, M_range)

    def predict_behavior(self) -> Dict:
        """
        Predict whether the map converges or diverges based on (D, gamma).

        Computes D and gamma at M=14 and uses the classify_map decision rule.

        Returns:
            Dict with 'D', 'gamma', 'prediction', 'confidence', 'details'
        """
        D = self.compute_D_M(14)
        gamma = self.compute_gamma_M(14)
        classification = classify_map(D, gamma)

        return {
            'D': D,
            'gamma': gamma,
            'prediction': classification['prediction'],
            'confidence': classification['confidence'],
            'details': classification['details'],
        }

    def verify_prediction(self, n_samples: int = 1000, max_steps: int = 10000) -> Dict:
        """
        Empirically verify the predicted behavior by running trajectories.

        Tests random odd integers and checks whether they converge (reach 1)
        or diverge (exceed a threshold).

        Args:
            n_samples: Number of random odd integers to test
            max_steps: Maximum number of iterations per trajectory

        Returns:
            Dict with 'n_samples', 'converged', 'diverged', 'undecided',
            'convergence_rate', 'max_value_seen'
        """
        threshold = 10**15
        converged = 0
        diverged = 0
        undecided = 0
        max_value_seen = 0

        for _ in range(n_samples):
            n = random.randrange(1, 10000, 2)  # random odd integer in [1, 9999]
            current = n
            reached_one = False
            exceeded = False

            for _ in range(max_steps):
                if current == 1:
                    reached_one = True
                    break
                if current > threshold:
                    exceeded = True
                    break
                current = self.f(current)
                if current > max_value_seen:
                    max_value_seen = current

            if reached_one:
                converged += 1
            elif exceeded:
                diverged += 1
            else:
                undecided += 1

        return {
            'n_samples': n_samples,
            'converged': converged,
            'diverged': diverged,
            'undecided': undecided,
            'convergence_rate': converged / n_samples if n_samples > 0 else 0.0,
            'max_value_seen': max_value_seen,
        }

    def analyze_countdown(self, n: int) -> Dict:
        """
        Decompose a Collatz orbit into countdown and reset phases.

        Args:
            n: Starting odd integer (must be odd)

        Returns:
            Dict with 'phases', 'countdown_fraction', 'mean_reset_ed',
            'total_steps', 'n_countdowns', 'n_resets'
        """
        phases = countdown_decomposition(n)

        countdowns = [p for p in phases if p['phase'] == 'countdown']
        resets = [p for p in phases if p['phase'] == 'reset']

        total_steps = sum(
            p.get('length', len(p['values']) - 1) for p in phases
        )

        countdown_steps = sum(p.get('length', len(p['values']) - 1) for p in countdowns)
        countdown_fraction = countdown_steps / total_steps if total_steps > 0 else 0.0

        reset_eds = [p['ed_end'] for p in resets if p['ed_end'] > 0]
        mean_reset_ed = float(np.mean(reset_eds)) if len(reset_eds) > 0 else 0.0

        return {
            'phases': phases,
            'countdown_fraction': countdown_fraction,
            'mean_reset_ed': mean_reset_ed,
            'total_steps': total_steps,
            'n_countdowns': len(countdowns),
            'n_resets': len(resets),
        }

    def verify_reset_distribution(self, M: int = 12, n_samples: int = 10000) -> Dict:
        """
        Verify that reset embedding depths follow Geo(1/2) distribution.

        Samples many odd n with ed(n)=1, applies the single-step map,
        and checks the output embedding depth distribution against Geo(1/2).

        Args:
            M: Modular resolution for sampling range
            n_samples: Number of samples to draw

        Returns:
            Dict with 'observed_counts', 'expected_counts', 'chi_squared',
            'p_value', 'geo_half_consistent'
        """
        output_eds = []
        attempts = 0
        max_attempts = n_samples * 20

        while len(output_eds) < n_samples and attempts < max_attempts:
            n = random.randrange(1, 2**M, 2)
            attempts += 1
            if embedding_depth(n) == 1:
                fn = single_step_map(n, self.q, self.c)
                if fn > 0:
                    output_eds.append(embedding_depth(fn))

        if len(output_eds) == 0:
            return {
                'observed_counts': {},
                'expected_counts': {},
                'chi_squared': float('inf'),
                'p_value': 0.0,
                'geo_half_consistent': False,
                'n_collected': 0,
            }

        # Count observed ed distribution
        from collections import Counter
        counts = Counter(output_eds)
        max_ed = max(counts.keys())
        n_collected = len(output_eds)

        # Expected Geo(1/2): P(ed=k) = (1/2)^k for k >= 1
        observed = {}
        expected = {}
        for k in range(1, max_ed + 1):
            observed[k] = counts.get(k, 0)
            expected[k] = n_collected * (0.5 ** k)

        # Bin tail (k > some cutoff) to avoid small expected counts
        cutoff = max(5, max_ed)
        obs_tail = sum(observed.get(k, 0) for k in range(cutoff, max_ed + 1))
        exp_tail = sum(expected.get(k, 0.0) for k in range(cutoff, max_ed + 1))
        # Also add the probability mass for k > max_ed
        exp_tail += n_collected * (0.5 ** cutoff)

        # Chi-squared statistic for bins 1..cutoff-1 + tail
        chi_sq = 0.0
        bins_used = 0
        for k in range(1, cutoff):
            if expected.get(k, 0) > 0:
                chi_sq += (observed.get(k, 0) - expected[k]) ** 2 / expected[k]
                bins_used += 1
        if exp_tail > 0:
            chi_sq += (obs_tail - exp_tail) ** 2 / exp_tail
            bins_used += 1

        # Degrees of freedom = bins - 1
        dof = max(bins_used - 1, 1)

        # p-value via scipy chi-squared survival function
        from scipy import stats as sp_stats
        p_value = float(sp_stats.chi2.sf(chi_sq, dof))

        geo_consistent = p_value > 0.01  # not rejected at 1% level

        return {
            'observed_counts': dict(observed),
            'expected_counts': {k: round(v, 2) for k, v in expected.items()},
            'chi_squared': round(chi_sq, 4),
            'p_value': round(p_value, 4),
            'geo_half_consistent': geo_consistent,
            'n_collected': n_collected,
        }

    def get_twelve_map_table(self) -> List[Dict]:
        """
        Return the pre-computed twelve map predictions reference table.

        Returns:
            List of dicts with map name, q, c, D_approx, gamma,
            cycles flag, and prediction.
        """
        return TWELVE_MAP_PREDICTIONS

    def verify_twelve_predictions(self, M: int = 12) -> Dict:
        """
        Verify the twelve map predictions by computing D and gamma
        for each map and comparing to the predicted behavior.

        Args:
            M: Modular resolution to use for computation

        Returns:
            Dict with 'results' (list of per-map dicts) and 'success_rate'
        """
        results = []
        correct = 0

        for entry in TWELVE_MAP_PREDICTIONS:
            q = entry['q']
            c = entry['c']
            has_cycles = entry['cycles']
            expected_prediction = entry['prediction']

            f = lambda n, _q=q, _c=c: single_step_map(n, _q, _c)
            D = compute_contraction_defect(f, M)
            gamma = compute_drift(f, M)
            classification = classify_map(D, gamma, has_cycles=has_cycles)

            match = classification['prediction'] == expected_prediction
            if match:
                correct += 1

            results.append({
                'map': entry['map'],
                'q': q,
                'c': c,
                'D_computed': round(D, 4),
                'D_expected': entry['D_approx'],
                'gamma_computed': round(gamma, 4),
                'gamma_expected': entry['gamma'],
                'prediction_computed': classification['prediction'],
                'prediction_expected': expected_prediction,
                'match': match,
            })

        n_maps = len(TWELVE_MAP_PREDICTIONS)
        return {
            'results': results,
            'success_rate': correct / n_maps if n_maps > 0 else 0.0,
            'correct': correct,
            'total': n_maps,
        }

    def piecewise_map(self, n: int, k: int, modulus: int = 16) -> int:
        """
        Piecewise 3/5 family map f_k(n).

        For 0 <= k <= 8, applies odd(5n+1) if n mod modulus is in
        the first k odd residues, otherwise odd(3n+1).

        Args:
            n: Odd positive integer
            k: Number of residue classes using 5n+1 (0=pure Collatz, 8=pure 5n+1)
            modulus: Residue class modulus (default 16)

        Returns:
            f_k(n) result
        """
        odd_residues = [r for r in range(1, modulus, 2)]
        five_residues = set(odd_residues[:k])

        if n % modulus in five_residues:
            return odd_part(5 * n + 1)
        else:
            return single_step_map(n, 3, 1)

    def gamma_interpolation(self, k_values: list = None, M: int = 12) -> Dict:
        """
        Compute gamma for piecewise 3/5 maps near gamma=1 transition.

        At k=0: pure Collatz (gamma=3/4)
        At k=8: pure 5n+1 (gamma=5/4)
        Intermediate: gamma interpolates through 1.

        Args:
            k_values: List of k values to evaluate (default: 0..8)
            M: Modular resolution for computation

        Returns:
            Dict with 'k_values', 'gamma_values', 'critical_k' (where gamma crosses 1)
        """
        if k_values is None:
            k_values = list(range(9))

        gamma_values = []
        for k in k_values:
            f = lambda n, k=k: self.piecewise_map(n, k)
            S_M = [n for n in range(1, 2**M, 2)]
            log_sum = 0.0
            count = 0
            for n in S_M:
                fn = f(n)
                if fn > 0 and n > 0:
                    log_sum += np.log2(fn / n)
                    count += 1
            gamma = 2.0 ** (log_sum / count) if count > 0 else None
            gamma_values.append(gamma)

        # Find critical k where gamma crosses 1
        critical_k = None
        for i in range(len(gamma_values) - 1):
            if gamma_values[i] is not None and gamma_values[i+1] is not None:
                if gamma_values[i] < 1.0 and gamma_values[i+1] >= 1.0:
                    # Linear interpolation
                    frac = (1.0 - gamma_values[i]) / (gamma_values[i+1] - gamma_values[i])
                    critical_k = k_values[i] + frac * (k_values[i+1] - k_values[i])
                    break

        return {
            'k_values': k_values,
            'gamma_values': gamma_values,
            'critical_k': critical_k,
            'transition_region': f'gamma crosses 1 near k={critical_k:.1f}' if critical_k else 'not found',
        }

    # ========== Domain-Specific Hooks ==========

    def _domain_specific_diagnose(self, data=None, **kwargs) -> Dict[str, Any]:
        """
        Diagnose number-theory-specific issues.

        Checks that the map is well-defined, D and gamma are in expected
        ranges, and the input data (if provided) is valid.
        """
        issues = []
        recommendations = []
        details = {}

        # Check map is well-defined on a sample
        try:
            test_val = self.f(3)
            if test_val <= 0:
                issues.append("Map produces non-positive output for n=3")
            details['f(3)'] = test_val
        except Exception as e:
            issues.append(f"Map evaluation failed: {e}")

        # Check D_M at moderate resolution
        try:
            D = self.compute_D_M(10)
            details['D_10'] = round(D, 4)
            if D < 1.0:
                issues.append(f"D_10 = {D:.4f} < 1.0, map appears injective (unexpected)")
                recommendations.append("Check map definition; most qn+c maps should be non-injective")
        except Exception as e:
            issues.append(f"Failed to compute D_10: {e}")

        # Check gamma at moderate resolution
        try:
            gamma = self.compute_gamma_M(10)
            details['gamma_10'] = round(gamma, 4)
            if gamma > 10.0:
                recommendations.append(f"gamma_10 = {gamma:.4f} is very large; map is strongly expanding")
        except Exception as e:
            issues.append(f"Failed to compute gamma_10: {e}")

        # Validate input data if provided
        if data is not None:
            if isinstance(data, (int, np.integer)):
                if int(data) % 2 == 0:
                    issues.append("Input integer is even; expected odd for Collatz-type analysis")
                    recommendations.append("Provide an odd integer as input")
                if int(data) <= 0:
                    issues.append("Input integer is non-positive")
            elif hasattr(data, '__len__'):
                if len(data) == 0:
                    issues.append("Input data is empty")
                    recommendations.append("Provide at least one odd positive integer")

        status = 'ok'
        if len(issues) > 0:
            status = 'warning'

        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'auto_fix': None,
            'details': details,
        }

    def _domain_specific_validate(self, data=None, **kwargs) -> Dict[str, bool]:
        """
        Validate number-theory-specific technique requirements.

        Checks:
        - map_well_defined: map produces valid output on sample inputs
        - D_above_one: contraction defect > 1 (non-injective)
        - gamma_finite: drift is finite and positive
        - resolution_stable: D stabilizes across resolutions
        """
        checks = {}

        # Map is well-defined
        try:
            results = [self.f(n) for n in [1, 3, 5, 7, 9]]
            checks['map_well_defined'] = all(r > 0 for r in results)
        except Exception:
            checks['map_well_defined'] = False

        # D > 1
        try:
            D = self.compute_D_M(10)
            checks['D_above_one'] = D > 1.0
        except Exception:
            checks['D_above_one'] = False

        # Gamma finite
        try:
            gamma = self.compute_gamma_M(10)
            checks['gamma_finite'] = np.isfinite(gamma) and gamma > 0
        except Exception:
            checks['gamma_finite'] = False

        # Resolution stability: check D at M=8,10,12
        try:
            D_vals = [self.compute_D_M(M) for M in [8, 10, 12]]
            cv = np.std(D_vals) / np.mean(D_vals) if np.mean(D_vals) > 0 else float('inf')
            checks['resolution_stable'] = cv < 0.2
        except Exception:
            checks['resolution_stable'] = False

        return checks

    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """
        Generate a human-readable explanation of number theory analysis results.

        Args:
            result: Result dictionary from predict_behavior, verify_prediction,
                    or other analysis methods.

        Returns:
            Markdown-formatted explanation string.
        """
        lines = ["# Number Theory Analysis Results", ""]

        if 'prediction' in result:
            lines.append(f"**Prediction:** {result['prediction']}")
        if 'confidence' in result:
            lines.append(f"**Confidence:** {result['confidence']}")
        if 'D' in result:
            lines.append(f"**Contraction Defect (D):** {result['D']:.4f}")
        if 'gamma' in result:
            lines.append(f"**Drift (gamma):** {result['gamma']:.4f}")
        if 'details' in result:
            lines.append(f"**Details:** {result['details']}")
        lines.append("")

        if 'convergence_rate' in result:
            lines.append("## Empirical Verification")
            lines.append(f"- Samples tested: {result.get('n_samples', 'N/A')}")
            lines.append(f"- Convergence rate: {result['convergence_rate']:.2%}")
            lines.append(f"- Converged: {result.get('converged', 'N/A')}")
            lines.append(f"- Diverged: {result.get('diverged', 'N/A')}")
            lines.append(f"- Undecided: {result.get('undecided', 'N/A')}")
            lines.append("")

        if 'success_rate' in result:
            lines.append("## Twelve Map Verification")
            lines.append(f"- Success rate: {result['success_rate']:.2%}")
            lines.append(f"- Correct: {result.get('correct', 'N/A')} / {result.get('total', 'N/A')}")
            lines.append("")

        if 'countdown_fraction' in result:
            lines.append("## Countdown Decomposition")
            lines.append(f"- Countdown fraction: {result['countdown_fraction']:.2%}")
            lines.append(f"- Mean reset ed: {result['mean_reset_ed']:.2f}")
            lines.append(f"- Total steps: {result['total_steps']}")
            lines.append(f"- Countdowns: {result['n_countdowns']}")
            lines.append(f"- Resets: {result['n_resets']}")
            lines.append("")

        lines.append("## Interpretation")
        lines.append("- D > 1 indicates the map is structurally non-injective (net compression)")
        lines.append("- gamma < 1 indicates the map is on average contracting")
        lines.append("- D > 1 and gamma < 1 together predict convergent behavior")
        lines.append("- gamma > 1 predicts divergent behavior regardless of D")

        return "\n".join(lines)


def _chi2_cdf(x: float, dof: int) -> float:
    """
    Approximate chi-squared CDF using series expansion of the
    regularized lower incomplete gamma function.

    This avoids a dependency on scipy for a simple goodness-of-fit test.
    """
    if x <= 0:
        return 0.0

    k = dof / 2.0
    x_half = x / 2.0

    # Series expansion: gamma_inc(k, x) = x^k * e^(-x) * sum_{n=0}^{inf} x^n / (k*(k+1)*...*(k+n))
    term = 1.0 / k
    total = term
    for n in range(1, 300):
        term *= x_half / (k + n)
        total += term
        if abs(term) < 1e-15 * abs(total):
            break

    result = total * (x_half ** k) * np.exp(-x_half)

    # Divide by Gamma(k) to get the regularized form
    # Use Stirling or math.gamma
    from math import gamma as math_gamma
    try:
        gamma_k = math_gamma(k)
    except (OverflowError, ValueError):
        return 0.5  # fallback

    cdf = result / gamma_k
    return max(0.0, min(1.0, cdf))
