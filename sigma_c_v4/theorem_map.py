"""
Theorem citation helper.

Cites the paper by LaTeX label (stable under renumbering), resolves the
current number via THEOREM_MAP.md (which lives at the sigmacore repo root).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import re


_MAP_PATH = Path(__file__).resolve().parent.parent / "THEOREM_MAP.md"
_CACHE: Optional[Dict[str, str]] = None


def _parse_theorem_map() -> Dict[str, str]:
    """Parse THEOREM_MAP.md tables into {label: number} dict."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    mapping: Dict[str, str] = {}
    if not _MAP_PATH.exists():
        _CACHE = mapping
        return mapping
    text = _MAP_PATH.read_text(encoding="utf-8")
    # Match table rows of form: | `label` | number | ... | ... |
    pattern = re.compile(r"^\|\s*`([\w:.-]+)`\s*\|\s*([\w.]+)\s*\|", re.M)
    for match in pattern.finditer(text):
        label, number = match.group(1), match.group(2)
        mapping[label] = number
    _CACHE = mapping
    return mapping


def cite(label: str, note: str = "") -> str:
    """
    Render a paper citation by label.

    Example:
        >>> cite("thm:trichotomy-geometric")
        'paper Thm 8.3 (thm:trichotomy-geometric)'

    If the label is missing from THEOREM_MAP.md, falls back to the label alone.
    """
    mapping = _parse_theorem_map()
    number = mapping.get(label)
    if number is None:
        base = f"paper [{label}]"
    else:
        # Heuristic: deduce kind from label prefix
        kind = {
            "thm": "Thm",
            "prop": "Prop",
            "def": "Def",
            "lem": "Lem",
            "cor": "Cor",
            "rem": "Rem",
            "obs": "Obs",
        }.get(label.split(":")[0], "")
        base = f"paper {kind} {number} ({label})"
    if note:
        return f"{base} — {note}"
    return base


def reload_map() -> None:
    """Force re-read of THEOREM_MAP.md (after paper renumbering)."""
    global _CACHE
    _CACHE = None
    _parse_theorem_map()
