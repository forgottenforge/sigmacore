"""
LaTeX Generator
===============
Generates professional LaTeX reports for Sigma-C analysis.

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

import os
from typing import Dict, Any, List

class LatexGenerator:
    """
    Auto-generates .tex files for publication.
    """
    
    def generate_report(self, 
                       title: str, 
                       author: str, 
                       abstract: str, 
                       sections: List[Dict[str, str]], 
                       filename: str):
        """
        Create a full LaTeX document.
        """
        content = [
            r"\documentclass{article}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath}",
            r"\usepackage{hyperref}",
            r"\title{" + title + "}",
            r"\author{" + author + "}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            abstract,
            r"\end{abstract}"
        ]
        
        for section in sections:
            content.append(r"\section{" + section['title'] + "}")
            content.append(section['content'])
            
        content.append(r"\end{document}")
        
        full_text = "\n".join(content)
        
        with open(f"{filename}.tex", "w") as f:
            f.write(full_text)
            
        print(f"Generated LaTeX report: {filename}.tex")
