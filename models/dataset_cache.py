from __future__ import annotations

"""
Shared helpers for caching `.npz` dataset generation.

My goals:
- I don't want to regenerate datasets if the final `.npz` already exists.
- I want to avoid copy-pasting "if not exists: generate" logic across notebooks/tasks.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional


def dataset_dir_for_task(*, repo_root: str | Path, task: str) -> Path:
    """
    Returns a dataset directory under `models/<task>/datasets/`.
    """
    rr = Path(repo_root)
    return rr / "models" / str(task) / "datasets"


def ensure_npz(
    *,
    out_path: str | Path,
    generator: Callable[..., Any],
    generator_kwargs: Optional[Dict[str, Any]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    If `out_path` exists, I return it (unless force=True).

    Otherwise I generate it once (serial only).
    """
    out_path = Path(out_path)
    if out_path.exists() and not bool(force):
        if verbose:
            print(f"[dataset] Using cached: {out_path}")
        return out_path

    if verbose:
        print(f"[dataset] Building: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw0 = dict(generator_kwargs or {})
    generator(output_path=out_path, **kw0)
    if verbose:
        print(f"[dataset] Wrote: {out_path}")
    return out_path


