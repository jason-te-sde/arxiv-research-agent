import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ROOT / "scripts" / "attribution-patterns.txt"


def test_patterns_file_has_the_attribution_shapes() -> None:
    lines = [ln for ln in PATTERNS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 5
    assert any("co-authored-by" in ln.lower() for ln in lines)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_hygiene_script_passes_on_the_repository() -> None:
    result = subprocess.run(  # noqa: S603
        ["bash", str(ROOT / "scripts" / "check-hygiene.sh")],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hygiene: ok" in result.stdout
