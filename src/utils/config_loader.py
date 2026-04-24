import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """carrega os yamls
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {p}\n"
            f"Expected location: {p.resolve()}"
        )
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Path):
    return load_yaml(path)