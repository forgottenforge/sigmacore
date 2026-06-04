"""
7-probe gallery: the full NISQ audit as a single composite figure.

Each panel shows one probe with:
  - the chi profile (paper convention or v4-pure, depending on availability)
  - the paper's sigma_c value as a vertical reference line
  - the v4-pure verdict (sigma_c / regime)
  - the paper-convention reproduction status (matches / mismatch)

Color coding:
  green  : both v4-pure and paper convention agree with paper anchor
  blue   : paper convention reproduces, v4-pure shows more structure (II)
  amber  : paper convention reproduces, v4-pure reports bottom (III)
  grey   : paper convention does NOT reproduce -- documented discrepancy
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sigma_c_v4.adapters.magnetic import ExperimentAudit


# Light theme palette (FUSE-aligned)
C_BG = "#ffffff"
C_CARD = "#f8f9fb"
C_TEXT = "#1a1a2e"
C_DIM = "#6b7280"
C_LIGHT = "#d1d5db"
C_GRID = "#e5e7eb"

C_AGREE = "#059669"        # green: both match
C_CONV_ONLY = "#2563eb"    # blue: convention matches, v4 sees more
C_CONV_BOTTOM = "#d97706"  # amber: convention matches, v4 reports bottom
C_DISAGREE = "#dc2626"     # red: convention does not match


def _panel_color(audit: ExperimentAudit) -> str:
    if audit.v4_result is None:
        return C_DISAGREE
    conv = audit.convention_result
    if conv is None:
        return C_DIM
    v4_matches = audit.matches_paper
    conv_matches = conv.matches_paper
    if v4_matches and conv_matches:
        return C_AGREE
    if conv_matches and isinstance(audit.v4_result.sigma_c, list):
        return C_CONV_ONLY
    if conv_matches and audit.v4_result.sigma_c is None:
        return C_CONV_BOTTOM
    return C_DISAGREE


def _panel_label(audit: ExperimentAudit) -> str:
    """Short title for the panel header."""
    eid = audit.experiment_id
    # strip parenthetical for headline
    base = eid.split(" (")[0]
    return base


def _panel_subtitle(audit: ExperimentAudit) -> str:
    if audit.v4_result is None:
        return "no v4 result"
    conv = audit.convention_result
    paper_sc = (f"{audit.paper_sigma_c:.3g}"
                if audit.paper_sigma_c is not None else "-")
    if conv is None:
        return f"paper {paper_sc} | conv: scipy missing"
    conv_sc = (f"{conv.sigma_c_convention:.3g}"
               if conv.sigma_c_convention is not None else "None")
    if conv.matches_paper:
        return f"paper {paper_sc} = conv {conv_sc} OK"
    return f"paper {paper_sc} != conv {conv_sc}"


def _draw_panel(ax, audit: ExperimentAudit) -> None:
    color = _panel_color(audit)

    # Plot the chi profile from the convention (which is the same view the
    # paper uses)
    if audit.convention_result is not None and audit.v4_result is not None:
        sigma = audit.v4_result._profile_sigma
        chi = audit.v4_result._profile_chi
    else:
        sigma, chi = None, None

    if sigma is not None and chi is not None and len(sigma) > 0:
        ax.plot(sigma, chi, color=color, lw=1.7)
        ax.fill_between(sigma, 0, chi, color=color, alpha=0.10)

        # Mark paper sigma_c if available
        if audit.paper_sigma_c is not None and audit.paper_sigma_c > 0:
            ax.axvline(audit.paper_sigma_c, color=C_TEXT, ls=":", lw=1.0,
                       alpha=0.7)

        # Mark paper-convention sigma_c
        if (audit.convention_result is not None
                and audit.convention_result.sigma_c_convention is not None
                and audit.convention_result.sigma_c_convention > 0):
            ax.axvline(
                audit.convention_result.sigma_c_convention,
                color=color, ls="--", lw=1.2,
            )

        # Use log scale where range allows
        positive_min = float(np.min(sigma))
        positive_max = float(np.max(sigma))
        if positive_min > 0 and positive_max / positive_min > 50:
            ax.set_xscale("log")

    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)
    ax.grid(True, alpha=0.30, color=C_GRID, lw=0.4)

    # Title and subtitle stacked above the axes (no overlap with chart)
    ax.text(
        0.0, 1.20, _panel_label(audit),
        transform=ax.transAxes,
        fontsize=10, color=C_TEXT, weight="bold", va="bottom",
    )
    ax.text(
        0.0, 1.05, _panel_subtitle(audit),
        transform=ax.transAxes,
        fontsize=8, color=color, weight="bold", va="bottom",
    )


def render_audit_gallery(
    audits: Dict[str, ExperimentAudit],
    *,
    save_to: Optional[Path] = None,
    title: str = "NISQ quantum-magnetism anchor audit -- seven probes under v4 discipline",
):
    """
    Render a composite 7-panel gallery of the audit results.

    Returns the matplotlib Figure (and saves to save_to if provided).
    """
    n = len(audits)
    # 4 columns x 2 rows looks clean for 7-8 probes
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(15.5, 4.8 * n_rows + 1.2), facecolor=C_BG)
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.35, hspace=0.95,
        left=0.05, right=0.97, top=0.84, bottom=0.08,
    )

    fig.text(
        0.05, 0.94, title,
        fontsize=15, color=C_TEXT, weight="bold",
    )
    fig.text(
        0.05, 0.91,
        "Vertical lines: dotted = paper sigma_c, dashed = paper convention "
        "reproduction. Color: green = both match, blue = convention matches "
        "+ v4 sees more, amber = convention matches + v4 = bottom, "
        "red = convention mismatch.",
        fontsize=8, color=C_DIM,
    )

    for i, (key, audit) in enumerate(audits.items()):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c], facecolor=C_CARD)
        _draw_panel(ax, audit)

    # Footer with summary count + branding
    n_v4 = sum(1 for a in audits.values() if a.matches_paper)
    n_conv = sum(
        1 for a in audits.values()
        if a.convention_result is not None
        and a.convention_result.matches_paper
    )
    fig.text(
        0.05, 0.06,
        f"Summary: v4-pure matches paper {n_v4}/{len(audits)}; "
        f"paper convention reproduces {n_conv}/{len(audits)}.   "
        f"backed by: paper Def 2.2 / Prop 4.1 / Thm 8.3.   "
        f"Convention = qmag2026:savgol5_dOdsigma.",
        fontsize=8.5, color=C_DIM,
    )
    fig.text(
        0.97, 0.06,
        "ForgottenForge - sigma_c v4",
        fontsize=8.5, color=C_LIGHT, ha="right",
    )

    if save_to is not None:
        fig.savefig(str(save_to), dpi=140, facecolor=C_BG,
                    bbox_inches="tight")
    return fig
