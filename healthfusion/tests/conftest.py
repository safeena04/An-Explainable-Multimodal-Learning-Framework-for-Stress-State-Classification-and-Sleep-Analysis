import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _exists(*parts) -> bool:
    return (ROOT.joinpath(*parts)).exists()


needs_processed = pytest.mark.skipif(not _exists("data", "processed", "qol_phase_features.csv"),
                                     reason="run python -m src.pipelines.run_preprocessing first")
needs_models = pytest.mark.skipif(not _exists("models", "registry.json"),
                                  reason="run the full pipeline (python -m src.pipelines.run_all) first")
