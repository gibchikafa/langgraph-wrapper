from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_root_module(module_name: str, file_name: str) -> ModuleType:
    root_dir = Path(__file__).resolve().parent.parent
    module_path = root_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not load {file_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_store_module = _load_root_module("langgraph_shim._store_impl", "store.py")
_previous_store_module = sys.modules.get("store")
sys.modules["store"] = _store_module
try:
    _app_module = _load_root_module("langgraph_shim._app_impl", "app.py")
finally:
    if _previous_store_module is None:
        sys.modules.pop("store", None)
    else:
        sys.modules["store"] = _previous_store_module

for name in dir(_app_module):
    if not name.startswith("_"):
        globals()[name] = getattr(_app_module, name)
