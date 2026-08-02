from __future__ import annotations

from importlib.metadata import distribution
from pathlib import Path
import re
import subprocess
import sys
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNTIME_DEPENDENCIES = {
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
}
ISOLATED_IMPORT_PROGRAM = """
from importlib.metadata import distribution
from pathlib import Path

working_directory = Path.cwd()
files_before = tuple(working_directory.iterdir())

import gridiron_spatial

installed = distribution("gridiron-spatial-intelligence")
assert gridiron_spatial.__name__ == "gridiron_spatial"
assert installed.metadata["Name"] == "gridiron-spatial-intelligence"
assert installed.version == "0.1.0"
assert tuple(working_directory.iterdir()) == files_before
print("isolated import passed")
"""


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_name(value: str) -> str:
    return _normalized_distribution_name(
        re.split(r"\s|[<>=!~;\[]", value, maxsplit=1)[0]
    )


def test_installed_package_metadata_and_import_are_data_free(
    tmp_path: Path,
) -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    declared = project["project"]
    installed = distribution(declared["name"])

    assert installed.metadata["Name"] == declared["name"]
    assert installed.version == declared["version"]
    assert REQUIRED_RUNTIME_DEPENDENCIES <= {
        _requirement_name(requirement)
        for requirement in declared["dependencies"]
    }

    completed = subprocess.run(
        [sys.executable, "-c", ISOLATED_IMPORT_PROGRAM],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "isolated import passed"
    assert tuple(tmp_path.iterdir()) == ()
