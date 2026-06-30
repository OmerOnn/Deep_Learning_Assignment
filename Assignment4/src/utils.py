from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist, and return it as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path