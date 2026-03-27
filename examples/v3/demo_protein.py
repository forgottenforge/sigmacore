#!/usr/bin/env python3
"""
Sigma-C Protein Stability Demo: Amyloid Disease Onset Prediction
================================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the protein stability adapter: thermodynamic sigma,
mutation analysis for TTR (transthyretin) amyloidosis, Spearman
correlation with disease onset, SOD1 negative control, and
mechanism classification.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sigma_c import Universe


def main():
    print("=" * 65)
    print("  PROTEIN STABILITY: AMYLOID DISEASE ONSET PREDICTION")
    print("=" * 65)

    # --- Create adapter via Universe factory ---
    prot = Universe.protein(protein_name='TTR')

    # ====================================================================
    # 1. Thermodynamic sigma at T_m = 1.0
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. THERMODYNAMIC CONTRACTION INDEX")
    print("-" * 65)
    # sigma = exp(-delta_G / (N * R * T))
    # At the melting temperature, delta_G ~ 0 so sigma ~ 1
    sigma_tm = prot.sigma_thermodynamic(delta_G=1.0, N=prot.N)
    print(f"  Protein       : TTR (Transthyretin)")
    print(f"  Residues (N)  : {prot.N}")
    print(f"  Temperature   : {prot.T} K")
    print(f"  R             : {prot.R:.3e} kcal/(mol*K)")
    print(f"  sigma(dG=1.0) : {sigma_tm:.6f}")
    print()
    print("  Sigma at various delta_G values:")
    print(f"  {'delta_G':>8}  {'sigma':>10}")
    print("  " + "-" * 20)
    for dg in [0.0, 1.0, 5.0, 10.0, 25.0]:
        s = prot.sigma_thermodynamic(dg, prot.N)
        print(f"  {dg:>8.1f}  {s:>10.6f}")

    # ====================================================================
    # 2. TTR mutation analysis
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. TTR MUTATION ANALYSIS (25 mutations)")
    print("-" * 65)

    mutations = prot.TTR_MUTATIONS
    result = prot.analyze_protein(mutations)

    # Show 5 representative mutations
    print("\n  Representative mutations:")
    print(f"  {'Name':<10} {'ddG':>6} {'sigma':>8} {'Onset_obs':>10} {'Onset_pred':>11}")
    print("  " + "-" * 50)

    # Pick 5 spread across the range
    indices = [0, 3, 11, 18, 23]  # T119M, V30M, L55P, A109T, D18G
    for i in indices:
        m = result['mutations'][i]
        obs_str = f"{m['onset_observed']}" if 'onset_observed' in m else "N/A"
        print(f"  {m['name']:<10} {m['delta_delta_G']:>6.1f} {m['sigma']:>8.4f} "
              f"{obs_str:>10} {m['onset_predicted']:>11.1f}")

    # Summary
    summary = result['summary']
    print(f"\n  Summary (n={summary['n_mutations']} mutations):")
    print(f"    sigma mean : {summary['sigma_mean']:.4f}")
    print(f"    sigma std  : {summary['sigma_std']:.4f}")
    print(f"    sigma range: [{summary['sigma_min']:.4f}, {summary['sigma_max']:.4f}]")

    # ====================================================================
    # 3. Spearman correlation: sigma vs onset
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. SPEARMAN CORRELATION (TTR)")
    print("-" * 65)
    corr = result['correlation']
    if 'sigma_vs_onset' in corr:
        sv = corr['sigma_vs_onset']
        print(f"  sigma vs onset:")
        print(f"    Spearman rho : {sv['spearman_rho']:.4f}")
        print(f"    p-value      : {sv['p_value']:.2e}")
        print(f"    n            : {sv['n']}")
        sig = "SIGNIFICANT" if sv['p_value'] < 0.05 else "not significant"
        print(f"    Result       : {sig} (alpha=0.05)")
        print(f"\n  Interpretation: negative rho means higher sigma -> earlier onset")

    if 'predicted_vs_observed_onset' in corr:
        pv = corr['predicted_vs_observed_onset']
        print(f"\n  predicted vs observed onset:")
        print(f"    Spearman rho : {pv['spearman_rho']:.4f}")
        print(f"    p-value      : {pv['p_value']:.2e}")

    # ====================================================================
    # 4. SOD1 negative control
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. SOD1 NEGATIVE CONTROL")
    print("-" * 65)
    sod1 = Universe.protein(protein_name='SOD1')
    sod1_result = sod1.analyze_protein(sod1.SOD1_MUTATIONS)
    sod1_corr = sod1_result['correlation']
    if 'sigma_vs_onset' in sod1_corr:
        sv1 = sod1_corr['sigma_vs_onset']
        print(f"  SOD1 sigma vs onset:")
        print(f"    Spearman rho : {sv1['spearman_rho']:.4f}")
        print(f"    p-value      : {sv1['p_value']:.4f}")
        sig1 = "SIGNIFICANT" if sv1['p_value'] < 0.05 else "NOT significant"
        print(f"    Result       : {sig1}")
        print(f"\n  SOD1 is gain-of-function: sigma analysis is expected")
        print(f"  to show weak/no correlation with onset age.")

    # ====================================================================
    # 5. Mechanism classification
    # ====================================================================
    print("\n" + "-" * 65)
    print("  5. MECHANISM CLASSIFICATION")
    print("-" * 65)

    ttr_mech = prot.classify_mechanism({
        'has_stable_fold': True,
        'delta_G': 1.0,
        'mutations_destabilizing': True,
    })
    print(f"  TTR:")
    print(f"    Mechanism       : {ttr_mech['mechanism']}")
    print(f"    Sigma applicable: {ttr_mech['sigma_applicable']}")
    print(f"    Rationale       : {ttr_mech['rationale'][:70]}...")

    sod1_mech = sod1.classify_mechanism({
        'has_stable_fold': True,
        'gain_of_function': True,
    })
    print(f"\n  SOD1:")
    print(f"    Mechanism       : {sod1_mech['mechanism']}")
    print(f"    Sigma applicable: {sod1_mech['sigma_applicable']}")
    print(f"    Rationale       : {sod1_mech['rationale'][:70]}...")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - sigma = exp(ddG / (N*R*T)) quantifies mutational destabilization")
    print("  - TTR: strong negative Spearman rho -> sigma predicts onset age")
    print("  - SOD1: weak/no correlation -> gain-of-function, outside scope")
    print("  - Mechanism classification guides when sigma analysis applies")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
