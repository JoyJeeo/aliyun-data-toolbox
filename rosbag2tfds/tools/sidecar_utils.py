"""Helpers for reading task sidecar metadata files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_sidecar(path: Optional[Path]) -> Dict:
    """Load JSON sidecar metadata if present."""
    if not path or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - IO error reporting
        print(f"  ⚠️ load_sidecar failed: {exc}")
        return {}


def marks_to_step_ranges(sidecar: Dict, num_steps: int) -> List[Dict]:
    """Convert normalized mark positions into concrete step ranges."""
    out: List[Dict] = []
    if not sidecar:
        return out
    marks = sidecar.get("marks") or sidecar.get("markList") or []
    for mark in marks:
        start = float(mark.get("startPosition", mark.get("markStartPosition", 0.0)) or 0.0)
        end = float(mark.get("endPosition", mark.get("markEndPosition", 1.0)) or 1.0)
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        i0 = int(round(start * max(0, num_steps - 1)))
        i1 = int(round(end * max(0, num_steps - 1)))
        if i1 < i0:
            i0, i1 = i1, i0
        out.append(
            {
                "step_start": i0,
                "step_end": i1,
                "skill": mark.get("skillAtomic") or mark.get("type") or "",
                "detail_zh": mark.get("skillDetail") or mark.get("detailZh") or "",
                "detail_en": mark.get("enDesc") or mark.get("enSkillDetail") or mark.get("detailEn") or "",
                "duration_s": mark.get("duration"),
                "raw": mark,
            }
        )
    return out


def clip_window_cover_all_marks(sidecar: Dict, num_steps: int) -> Optional[tuple[int, int]]:
    """Span the earliest start to latest end across all marks."""
    marks = sidecar.get("marks") or sidecar.get("markList") or []
    if not marks:
        return None
    starts = []
    ends = []
    for mark in marks:
        try:
            starts.append(float(mark.get("startPosition", 0.0)))
            ends.append(float(mark.get("endPosition", 1.0)))
        except Exception:
            continue
    if not starts or not ends:
        return None

    def _pos_to_idx01(x: float) -> int:
        x = max(0.0, min(1.0, float(x)))
        return int(round(x * max(0, num_steps - 1)))

    i0 = _pos_to_idx01(min(starts))
    i1 = _pos_to_idx01(max(ends))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1
