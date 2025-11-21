"""
Publication Visualizer
======================
Generates journal-quality figures for Sigma-C results.
Supports multi-panel layouts, automatic annotation, and high-DPI export.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os

class PublicationVisualizer:
    """
    Creates professional, publication-ready visualizations.
    """
    
    def __init__(self, style: str = 'nature'):
        self.style = style
        self._set_style()
        
    def _set_style(self):
        """Apply publication-specific style settings."""
        # Reset to defaults first
        plt.style.use('default')
        
        if self.style == 'nature':
            rcParams['font.family'] = 'sans-serif'
            rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
            rcParams['font.size'] = 7
            rcParams['axes.linewidth'] = 0.5
            rcParams['xtick.major.width'] = 0.5
            rcParams['ytick.major.width'] = 0.5
            rcParams['figure.dpi'] = 300
        elif self.style == 'prl': # Physical Review Letters
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            rcParams['font.size'] = 10
            
    def create_multi_panel_figure(self, 
                                  panels: List[Dict[str, Any]], 
                                  layout: Tuple[int, int] = (2, 2),
                                  figsize: Tuple[float, float] = (8.5, 11)) -> plt.Figure:
        """
        Create a figure with multiple panels.
        
        Args:
            panels: List of dicts, each containing 'plot_func', 'data', 'title', 'xlabel', 'ylabel'.
            layout: (rows, cols)
            figsize: (width, height) in inches
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(layout[0], layout[1])
        
        for i, panel in enumerate(panels):
            if i >= layout[0] * layout[1]:
                break
                
            row = i // layout[1]
            col = i % layout[1]
            ax = fig.add_subplot(gs[row, col])
            
            # Execute the plotting function
            if 'plot_func' in panel:
                panel['plot_func'](ax, panel.get('data'))
                
            # Styling
            ax.set_title(panel.get('title', ''), fontweight='bold')
            ax.set_xlabel(panel.get('xlabel', ''))
            ax.set_ylabel(panel.get('ylabel', ''))
            
            # Add panel label (a, b, c...)
            label = chr(97 + i) # 'a', 'b', 'c'...
            ax.text(-0.1, 1.1, f"({label})", transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', va='top', ha='right')
            
            # Add annotations if present
            if 'annotations' in panel:
                for ann in panel['annotations']:
                    ax.text(ann['x'], ann['y'], ann['text'], 
                           transform=ax.transAxes if ann.get('relative', False) else ax.transData,
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

        plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = ['png', 'pdf']):
        """Save figure in multiple formats."""
        for fmt in formats:
            full_path = f"{filename}.{fmt}"
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {full_path}")

    @staticmethod
    def plot_sigma_c_landscape(ax, data: Dict[str, Any]):
        """Standard plot for sigma_c landscape."""
        x = data['x']
        y = data['y']
        sigma_c = data.get('sigma_c')
        
        ax.plot(x, y, 'b-', linewidth=1.5, label='Response')
        
        if sigma_c:
            ax.axvline(sigma_c, color='r', linestyle='--', alpha=0.8, label=r'$\sigma_c$')
            # Add shaded region for confidence if available
            if 'sigma_c_std' in data:
                std = data['sigma_c_std']
                ax.axvspan(sigma_c - std, sigma_c + std, color='r', alpha=0.2)
                
        ax.legend()
        ax.grid(True, alpha=0.3)
