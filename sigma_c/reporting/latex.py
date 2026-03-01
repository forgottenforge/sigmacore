"""
Sigma-C LaTeX Report Generator
================================
Copyright (c) 2025 ForgottenForge.xyz

Generates publication-quality LaTeX reports from Sigma-C analysis results.
Supports tables, figures, and BibTeX references.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from typing import Dict, Any, List, Optional


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in user-provided text."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


class LatexGenerator:
    """
    Generate LaTeX reports for Sigma-C analysis results.

    Usage:
        from sigma_c.reporting.latex import LatexGenerator

        gen = LatexGenerator()
        gen.generate_report(
            title="Criticality Analysis of 2D Ising Model",
            author="Researcher Name",
            abstract="We analyze the phase transition...",
            sections=[
                {'title': 'Introduction', 'content': 'The Ising model...'},
                {'title': 'Results', 'content': 'We find sigma_c = 2.269...'},
            ],
            filename='report'
        )
    """

    def generate_report(self, title: str, author: str, abstract: str,
                        sections: List[Dict[str, str]], filename: str,
                        packages: Optional[List[str]] = None) -> str:
        """
        Create a full LaTeX document.

        Args:
            title: Document title.
            author: Author name(s).
            abstract: Abstract text.
            sections: List of {'title': ..., 'content': ...} dicts.
            filename: Output filename (without .tex extension).
            packages: Additional LaTeX packages to include.

        Returns:
            Path to the generated .tex file.
        """
        safe_title = _latex_escape(title)
        safe_author = _latex_escape(author)

        content = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{hyperref}",
            r"\usepackage{booktabs}",
            r"\usepackage{siunitx}",
        ]

        for pkg in (packages or []):
            content.append(r"\usepackage{" + pkg + "}")

        content.extend([
            r"\title{" + safe_title + "}",
            r"\author{" + safe_author + "}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            abstract,
            r"\end{abstract}",
        ])

        for section in sections:
            content.append(r"\section{" + _latex_escape(section['title']) + "}")
            content.append(section['content'])

        content.append(r"\end{document}")

        full_text = "\n\n".join(content)

        out_path = f"{filename}.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        return out_path

    def generate_results_table(self, results: List[Dict[str, Any]],
                               columns: Optional[List[str]] = None,
                               caption: str = "Sigma-C Analysis Results") -> str:
        """
        Generate a LaTeX booktabs table from analysis results.

        Args:
            results: List of result dictionaries.
            columns: Column keys to include (default: all keys from first result).
            caption: Table caption.

        Returns:
            LaTeX table string.
        """
        if not results:
            return ""

        if columns is None:
            columns = list(results[0].keys())

        header = " & ".join(_latex_escape(c) for c in columns)
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{" + "l" * len(columns) + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]

        for row in results:
            cells = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(_latex_escape(str(val)))
            lines.append(" & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{" + _latex_escape(caption) + "}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_figure(self, image_path: str, caption: str,
                        label: str = "fig:result", width: str = "0.8") -> str:
        """
        Generate a LaTeX figure inclusion.

        Args:
            image_path: Path to the image file.
            caption: Figure caption.
            label: LaTeX label for cross-referencing.
            width: Width as fraction of textwidth.

        Returns:
            LaTeX figure string.
        """
        return "\n".join([
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=" + width + r"\textwidth]{" + image_path + "}",
            r"\caption{" + _latex_escape(caption) + "}",
            r"\label{" + label + "}",
            r"\end{figure}",
        ])
