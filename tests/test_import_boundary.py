import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_does_not_import_locata():
    code = (
        "import sys; import ssl_doa_baselines.api; "
        "bad=[m for m in ('engine','data','losses','models') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True,
                         env={"PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_public_api_symbols():
    from ssl_doa_baselines import api

    for name in ("build_model", "load_model", "predict", "predict_native", "fit"):
        assert callable(getattr(api, name))
