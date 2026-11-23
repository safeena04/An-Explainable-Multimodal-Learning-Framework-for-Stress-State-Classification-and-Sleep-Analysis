"""Extract the two PhysioNet archives into data/raw/ with the layout the code expects.

Usage (from the project root):
    python scripts/setup_data.py --mmash path/to/mmash-1.0.0.zip \
                                 --qol path/to/qol_stress-...-1.0.0.zip

MMASH ships as an outer zip that contains a second zip (MMASH.zip). Both levels
are handled. Nothing is modified: files are only extracted and copied.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract while refusing paths that escape the destination (zip-slip)."""
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def setup_mmash(archive: Path) -> None:
    out = RAW / "mmash"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, tmp)
        inner = next(tmp.rglob("MMASH.zip"), None)
        if inner is not None:
            with zipfile.ZipFile(inner) as zf:
                _safe_extract(zf, tmp / "inner")
            search_root = tmp / "inner"
        else:
            search_root = tmp
        data_paper = next(search_root.rglob("DataPaper"), None)
        if data_paper is None:
            sys.exit("Could not find the DataPaper folder inside the MMASH archive.")
        if (out / "DataPaper").exists():
            shutil.rmtree(out / "DataPaper")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(data_paper, out / "DataPaper")
    users = sorted(p.name for p in (out / "DataPaper").glob("user_*"))
    print(f"MMASH: {len(users)} participant folders -> {out / 'DataPaper'}")


def setup_qol(archive: Path) -> None:
    out = RAW / "qol_stress"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, tmp)
        ecg = next(tmp.rglob("ecg.csv"), None)
        if ecg is None:
            sys.exit("Could not find ecg.csv inside the QoL_Stress archive.")
        out.mkdir(parents=True, exist_ok=True)
        for f in ecg.parent.glob("*.csv"):
            shutil.copy2(f, out / f.name)
        desc = next(tmp.rglob("data_description.csv"), None)
        if desc is not None:
            shutil.copy2(desc, out / "data_description.csv")
    print(f"QoL_Stress: {len(list(out.glob('*.csv')))} CSV files -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmash", type=Path, required=True)
    ap.add_argument("--qol", type=Path, required=True)
    args = ap.parse_args()
    setup_mmash(args.mmash)
    setup_qol(args.qol)


if __name__ == "__main__":
    main()
