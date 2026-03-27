#!/usr/bin/env python3
"""
Sigma-C Linguistics Demo: Etymological Depth and Semantic Change
================================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Demonstrates the linguistics adapter: etymological depth (ED)
lookup, correlation analysis, fixed-point test, transparency
paradox, and German anchor test.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sigma_c import Universe


def main():
    print("=" * 65)
    print("  LINGUISTICS: ETYMOLOGICAL DEPTH & SEMANTIC CHANGE")
    print("=" * 65)

    # --- Create adapter via Universe factory ---
    ling = Universe.linguistics()

    # ====================================================================
    # 1. Etymological depth for sample words
    # ====================================================================
    print("\n" + "-" * 65)
    print("  1. ETYMOLOGICAL DEPTH (ED) FOR SAMPLE WORDS")
    print("-" * 65)
    sample_words = ['I', 'beautiful', 'algorithm', 'water', 'government',
                    'unfortunately', 'dog', 'trivial']
    print(f"  {'Word':<18} {'ED':>4}  {'Change':>8}")
    print("  " + "-" * 34)
    for word in sample_words:
        ed = ling.etymological_depth(word)
        change = ling.semantic_change(word)
        ed_str = str(ed) if ed is not None else "N/A"
        ch_str = f"{change:.4f}" if change is not None else "N/A"
        print(f"  {word:<18} {ed_str:>4}  {ch_str:>8}")

    print("\n  Key examples: I(ED=1), beautiful(ED=3), algorithm(ED=4)")

    # ====================================================================
    # 2. Correlation analysis
    # ====================================================================
    print("\n" + "-" * 65)
    print("  2. CORRELATION ANALYSIS (ED vs Semantic Change)")
    print("-" * 65)
    corr = ling.correlation_analysis()
    print(f"  Pearson  r   = {corr['pearson_r']:.4f}  (p = {corr['pearson_p']:.2e})")
    print(f"  Spearman rho = {corr['spearman_rho']:.4f}  (p = {corr['spearman_p']:.2e})")
    print(f"  N            = {corr['n']}")
    sig = "SIGNIFICANT" if corr['spearman_p'] < 0.05 else "not significant"
    print(f"  Result       : {sig} positive correlation")
    print()
    print("  Interpretation: higher ED -> more semantic change over time")

    # ====================================================================
    # 3. Fixed-point test: ED=1 words are semantic invariants
    # ====================================================================
    print("\n" + "-" * 65)
    print("  3. FIXED-POINT TEST (ED=1 vs ED>1)")
    print("-" * 65)
    fp = ling.fixed_point_test()
    print(f"  Mean change (ED=1)  : {fp['mean_ed1']:.4f}")
    print(f"  Mean change (ED>1)  : {fp['mean_ed_gt1']:.4f}")
    print(f"  Welch t-statistic   : {fp['t_statistic']:.4f}")
    print(f"  p-value             : {fp['p_value']:.2e}")
    print(f"  Cohen's d           : {fp['cohens_d']:.4f}")
    sig2 = "YES" if fp['p_value'] < 0.05 else "NO"
    print(f"  Significant (a=0.05): {sig2}")
    print()
    print("  ED=1 words (primes) show LESS semantic change -> semantic invariants")

    # ====================================================================
    # 4. Transparency paradox: transparent words change MORE
    # ====================================================================
    print("\n" + "-" * 65)
    print("  4. TRANSPARENCY PARADOX (ED >= 2 words)")
    print("-" * 65)
    tr = ling.transparency_effect()
    print(f"  Mean change (transparent) : {tr['mean_transparent']:.4f}")
    print(f"  Mean change (opaque)      : {tr['mean_opaque']:.4f}")
    print(f"  Direction                 : {tr['direction']}")
    print(f"  Cohen's d                 : {tr['cohens_d']:.4f}")
    print(f"  p-value                   : {tr['p_value']:.4f}")
    print()
    if 'transparent > opaque' in tr['direction']:
        print("  Paradox confirmed: transparent words change MORE in English")
        print("  (morphological transparency exposes components to independent drift)")
    else:
        print("  Transparent and opaque words show different change rates")

    # ====================================================================
    # 5. German anchor test: mirror effect
    # ====================================================================
    print("\n" + "-" * 65)
    print("  5. GERMAN ANCHOR TEST (P < T < O mirror effect)")
    print("-" * 65)
    ga = ling.german_anchor_test()
    print(f"  ANOVA F-statistic : {ga['F_statistic']:.4f}")
    print(f"  ANOVA p-value     : {ga['p_value']:.2e}")
    print(f"  Mirror effect     : {ga['mirror_effect']}")
    print()
    print("  Pairwise comparisons:")
    for pair, vals in ga['pairwise'].items():
        sig3 = "*" if vals['p_value'] < 0.05 else " "
        print(f"    {pair:<8}  t = {vals['t_statistic']:>7.3f}  "
              f"p = {vals['p_value']:.4f} {sig3}")
    print()
    if ga['mirror_effect']:
        print("  Mirror effect confirmed: P(primes) < T(transparent) < O(opaque)")
        print("  In German, transparent words change LESS than opaque (reversed!)")
        print("  German's compositional morphology anchors transparent compounds")
    else:
        print("  Mirror effect not observed in this run")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("  - ED=1 words are semantic invariants (fixed-point behavior)")
    print("  - Positive correlation: higher ED -> more semantic change")
    print("  - English paradox: transparent words change MORE")
    print("  - German mirror: transparent words change LESS (anchoring)")
    print("  - Cross-linguistic difference reveals morphological effects")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
