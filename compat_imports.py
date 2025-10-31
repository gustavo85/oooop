"""
Compat layer to map base module names (without _Version suffixes) to the actual
versioned files present in the project, without renaming files on disk.

Usage:
    import compat_imports  # must be imported before importing base names
    from config_loader import ConfigurationManager, GameProfile
    from system_optimizers import AdvancedTimerManager, ...
"""

import sys
import importlib.util
from pathlib import Path
from typing import Optional

_BASE_TO_PREFIX = {
    "config_loader": "config_loader",
    "system_optimizers": "system_optimizers",
    "directx_optimizer": "directx_optimizer",
    "network_optimizer": "network_optimizer",
    "session_manager": "session_manager",
}

def _find_best_match(prefix: str) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [p for p in here.iterdir() if p.is_file() and p.name.lower().startswith(prefix.lower()) and p.suffix.lower() == ".py"]
    if not candidates:
        return None
    # Choose the most recently modified candidate (best guess for "latest")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _load_as_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

for base_name, prefix in _BASE_TO_PREFIX.items():
    if base_name in sys.modules:
        continue
    match = _find_best_match(prefix)
    if match:
        try:
            _load_as_module(base_name, match)
        except Exception as e:
            pass