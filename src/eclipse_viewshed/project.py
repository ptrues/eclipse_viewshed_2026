"""Locate files stored at the repository root."""

from pathlib import Path


def repository_root() -> Path:
    """Return the repository containing the installed editable package."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "The eclipse-viewshed package must be installed from its repository "
            "with `python -m pip install --no-deps -e .`."
        )
    return root
