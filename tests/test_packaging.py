from __future__ import annotations

import importlib
from importlib.metadata import distribution
from pathlib import Path
import re
import sys
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNTIME_DEPENDENCIES = {
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
}


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_name(value: str) -> str:
    return _normalized_distribution_name(
        re.split(r"\s|[<>=!~;\[]", value, maxsplit=1)[0]
    )


def test_installed_package_metadata_and_import_are_data_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    declared = project["project"]
    installed = distribution(declared["name"])

    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.iterdir())
    modules_before = set(sys.modules)
    package = importlib.import_module("gridiron_spatial")

    assert package.__name__ == "gridiron_spatial"
    assert installed.metadata["Name"] == declared["name"]
    assert installed.version == declared["version"]
    assert REQUIRED_RUNTIME_DEPENDENCIES <= {
        _requirement_name(requirement)
        for requirement in declared["dependencies"]
    }
    assert {
        name
        for name in set(sys.modules) - modules_before
        if name.startswith("gridiron_spatial")
    } == {"gridiron_spatial"}
    assert tuple(tmp_path.iterdir()) == files_before
