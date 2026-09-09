from __future__ import annotations

import ast
import json
from typing import Any, Dict


def parse_objects_bboxes(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return {"image_bboxes": []}

    return {"image_bboxes": []}
