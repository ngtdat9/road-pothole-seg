"""The package imports only numpy, opencv and onnxruntime besides the standard library."""

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "road_pothole_seg"
ALLOWED = {"numpy", "cv2", "onnxruntime", "road_pothole_seg"}


def _top_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_package_imports_only_allowed_modules() -> None:
    for path in sorted(PACKAGE.glob("*.py")):
        extra = _top_level_imports(path) - ALLOWED - set(sys.stdlib_module_names)
        assert not extra, f"{path.name} imports {sorted(extra)}"
