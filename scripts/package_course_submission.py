"""Create a Windows Explorer compatible course-submission ZIP."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "traffic-congestion-analysis-system-course-submission.zip"

EXCLUDED_DIRS = {".git", ".venv", "dist", "__pycache__", ".pytest_cache", ".tmp-pytest"}
EXCLUDED_FILES = {
    Path("data/processed/model_dataset/train_features.parquet"),
    Path("data/processed/model_dataset/validation_features.parquet"),
    Path("data/processed/model_dataset/test_features.parquet"),
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative in EXCLUDED_FILES or path.suffix == ".pyc":
        return False
    return not any(part in EXCLUDED_DIRS or part.startswith(".tmp-") for part in relative.parts)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()

    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and should_include(path))
    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo.from_file(path, arcname=relative)
            info.create_system = 0
            info.external_attr = 0
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
