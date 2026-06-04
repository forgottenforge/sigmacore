"""
Sigma-c Card — publication-quality visualization, FUSE-inspired light theme.

Sigma-c-specific semantics, not FUSE's tipping-point semantics:
  - Regime I  (single mode)        → blue (clean, falsifiable reading)
  - Regime II (multi-mode)         → amber (vector-valued sigma_c)
  - Regime III (no peak / floor)   → grey (sigma_c = ⊥ as positive output)
  - rho_star fitted (not analytic) → red flag in footer
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sigma_c_v4.result import Result


# ---------------------------------------------------------------------------
# Color palette — light theme, FUSE-aligned
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_CARD = "#f8f9fb"
C_TEXT = "#1a1a2e"
C_DIM = "#6b7280"
C_LIGHT = "#d1d5db"
C_GRID = "#e5e7eb"

C_REGIME_I = "#2563eb"     # blue — single mode, falsifiable
C_REGIME_II = "#d97706"    # amber — multi-mode
C_REGIME_III = "#6b7280"   # grey — undefined / scale-invariant
C_FLOOR = "#7c3aed"        # purple — operational floor triggered
C_FITTED = "#dc2626"       # red — fitted rho_star, exploratory
C_OK = "#059669"           # green — analytic rho_star, falsifiable


def _regime_color(result: Result) -> str:
    if result.regime.operational_floor_triggered:
        return C_FLOOR
    if result.sigma_c is None:
        return C_REGIME_III
    if isinstance(result.sigma_c, list):
        return C_REGIME_II
    return C_REGIME_I


def _regime_text(result: Result) -> str:
    if result.regime.operational_floor_triggered:
        return "REGIME III (operational floor)"
    g = result.regime.geometric
    if g == "I_geom":
        return "REGIME I (single mode)"
    if g == "II_geom":
        return "REGIME II (multi-mode)"
    return "REGIME III (no interior scale)"


def render(
    result: Result,
    sigma: np.ndarray,
    chi: np.ndarray,
    *,
    title: str = "",
    x_name: str = "σ (resolution)",
    y_name: str = "χ_O(σ)",
    save_to: Optional[Union[str, Path]] = None,
    figsize=(11.0, 6.0),
):
    """
    Render a sigma-c card for the result. Returns the matplotlib Figure
    (and saves to save_to if provided).
    """
    fig = plt.figure(figsize=figsize, facecolor=C_BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[2.2, 1.0], height_ratios=[1.0, 0.22],
        hspace=0.55, wspace=0.10,
        left=0.07, right=0.97, top=0.82, bottom=0.08,
    )

    # ===== Title bar (above the axes, two clean lines) =====
    color = _regime_color(result)
    fig.text(
        0.07, 0.93, title or "sigma_c v4 — disciplined-reader output",
        fontsize=15, color=C_TEXT, weight="bold",
    )
    fig.text(
        0.07, 0.875, _regime_text(result),
        fontsize=11, color=color, weight="bold",
    )

    # ===== Main panel: chi_O profile =====
    ax = fig.add_subplot(gs[0, 0], facecolor=C_CARD)
    ax.plot(sigma, chi, color=color, lw=2.2, label="χ_O(σ)")
    ax.fill_between(sigma, 0, chi, color=color, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlabel(x_name, color=C_TEXT, fontsize=11)
    ax.set_ylabel(y_name, color=C_TEXT, fontsize=11)
    ax.tick_params(colors=C_DIM, labelsize=9)
    ax.grid(True, alpha=0.35, color=C_GRID, ls="-", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)

    # Mark sigma_c (scalar or vector)
    if isinstance(result.sigma_c, list):
        for s in result.sigma_c:
            ax.axvline(s, color=color, ls="--", lw=1.2, alpha=0.85)
            ax.text(
                s, ax.get_ylim()[1] * 0.92,
                f"σ_c = {s:.3g}",
                color=color, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=color, lw=0.8),
            )
    elif result.sigma_c is not None:
        ax.axvline(result.sigma_c, color=color, ls="--", lw=1.5)
        ax.text(
            result.sigma_c, ax.get_ylim()[1] * 0.92,
            f"σ_c = {result.sigma_c:.3g}",
            color=color, ha="center", fontsize=10, weight="bold",
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor="white", edgecolor=color, lw=1.0),
        )
    else:
        ax.text(
            0.5, 0.5, "σ_c = ⊥\n(no interior peak)",
            transform=ax.transAxes,
            color=C_DIM, fontsize=18, ha="center", va="center",
            weight="bold", alpha=0.7,
        )

    # ===== Right panel: stats =====
    ax2 = fig.add_subplot(gs[0, 1], facecolor=C_BG)
    ax2.axis("off")
    y = 1.0
    line_h = 0.085

    def line(label: str, value: str, value_color: str = C_TEXT,
             label_color: str = C_DIM):
        nonlocal y
        ax2.text(0.02, y, label, fontsize=9, color=label_color,
                 transform=ax2.transAxes)
        ax2.text(0.98, y, value, fontsize=10.5, color=value_color, weight="bold",
                 ha="right", transform=ax2.transAxes)
        y -= line_h

    # sigma_c
    if result.sigma_c is None:
        line("σ_c", "⊥", C_REGIME_III)
    elif isinstance(result.sigma_c, list):
        line("σ_c", f"{len(result.sigma_c)}-vec", _regime_color(result))
    else:
        line("σ_c", f"{result.sigma_c:.4g}", _regime_color(result))

    # tau
    if result.tau is not None:
        line("τ = σ_c / ρ_⋆", f"{result.tau:.4g}", C_TEXT)
    else:
        line("τ", "—", C_DIM)

    # rho_star with provenance flag (Enforcement 2)
    if result.rho_star is not None:
        line(
            "ρ_⋆",
            f"{result.rho_star:.4g}",
            C_OK if result.falsifiable else C_FITTED,
        )
        line(
            "source",
            result.rho_star_source,
            C_OK if result.falsifiable else C_FITTED,
        )
    else:
        line("ρ_⋆", "—", C_DIM)
        line("source", "—", C_DIM)

    # gamma_O
    if result.gamma_O is not None:
        gc = C_FITTED if result.gamma_O < 0.1 else C_OK
        line("γ_O (SOC)", f"{result.gamma_O:.3g}", gc)
    else:
        line("γ_O", "—", C_DIM)

    # framework
    if result.framework is not None:
        line("framework", result.framework.value, C_TEXT)
    else:
        line("framework", "—  (dominant-scale)", C_DIM)

    # Falsifiability badge
    y -= 0.02
    if result.falsifiable and result.sigma_c is not None:
        ax2.text(
            0.5, y, "✓ FALSIFIABLE  (analytic ρ_⋆)",
            transform=ax2.transAxes,
            color=C_OK, fontsize=9.5, weight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor=C_OK, lw=1.2),
        )
    elif result.sigma_c is not None:
        ax2.text(
            0.5, y, "⚠ FITTED — exploratory",
            transform=ax2.transAxes,
            color=C_FITTED, fontsize=9.5, weight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor=C_FITTED, lw=1.2),
        )

    # ===== Footer: notes + citations =====
    foot_ax = fig.add_subplot(gs[1, :], facecolor=C_BG)
    foot_ax.axis("off")
    foot_lines = []
    if result.notes:
        for note in result.notes[:2]:
            foot_lines.append(f"• {note}")
    if result.citations:
        # cite up to 3 key labels
        from sigma_c_v4.theorem_map import cite
        cite_str = " · ".join(cite(c) for c in result.citations[:3])
        foot_lines.append(f"backed by: {cite_str}")
    foot_text = "\n".join(foot_lines) if foot_lines else ""
    foot_ax.text(
        0.0, 0.30, foot_text,
        transform=foot_ax.transAxes,
        color=C_DIM, fontsize=8.5, va="top",
    )
    foot_ax.text(
        1.0, 0.30, "ForgottenForge · sigma_c v4",
        transform=foot_ax.transAxes,
        color=C_LIGHT, fontsize=8.0, va="top", ha="right",
    )

    if save_to is not None:
        fig.savefig(str(save_to), dpi=150, facecolor=C_BG,
                    bbox_inches="tight")
    return fig


def _result_card_method(self, save_to: Union[str, Path], **kwargs):
    """Bound on Result via monkey-patch in api.__init__."""
    # Re-compute chi for display from stored smoothing recipe (or refuse)
    raise NotImplementedError(
        "Use sigma_c_v4.card.render(result, sigma, chi) — Result does not "
        "currently retain the chi profile. Pass sigma and chi from the same "
        "analyze call."
    )


# Convenience monkey-patch (optional): result.card(path)
def attach_card_method():
    Result.card = _result_card_method
