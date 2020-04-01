from pathlib import Path
from typing import Dict

import json5 as json

from hyperflow import init_data


def is_hdl_bottom(key, value):
    if isinstance(key, str) and key.startswith("__"):
        return True
    if isinstance(value, dict) and "_type" in value:
        return True
    if not isinstance(value, dict):
        return True
    return False


def get_hdl_db(path: str) -> Dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    else:
        print("warn")
        return get_default_hdl_db()


def get_default_hdl_db() -> Dict:
    return json.loads((Path(init_data.__file__).parent / f"hdl_db.json").read_text())
