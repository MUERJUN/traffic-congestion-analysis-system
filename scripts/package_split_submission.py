"""Create six Windows-compatible submission ZIPs, each below 20 MiB."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "split"
CHUNK_SIZE = 13 * 1024 * 1024
LARGE_FILES = {
    Path("data/raw/metr-la.h5"),
    Path("data/interim/metr_la_eda_long.parquet"),
    Path("artifacts/models/random_forest.joblib"),
}
DERIVED_MATRICES = {
    Path("data/processed/model_dataset/train_features.parquet"),
    Path("data/processed/model_dataset/validation_features.parquet"),
    Path("data/processed/model_dataset/test_features.parquet"),
}
EXCLUDED_DIRS = {".git", ".venv", "dist", "__pycache__", ".pytest_cache", ".tmp-pytest", "submission_parts"}


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 0
    info.external_attr = 0
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def open_zip(name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(OUT / name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)


def add_path(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zip_info(arcname)
    with path.open("rb") as source, archive.open(info, "w") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def core_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative in LARGE_FILES or relative in DERIVED_MATRICES or path.suffix == ".pyc":
        return False
    return not any(part in EXCLUDED_DIRS or part.startswith(".tmp-") for part in relative.parts)


def write_chunk_package(package_name: str, source: Path, part_name: str, offset: int) -> None:
    with source.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(CHUNK_SIZE)
    if not payload:
        raise ValueError(f"分片为空：{part_name}")
    with open_zip(package_name) as archive:
        archive.writestr(zip_info(f"submission_parts/{part_name}"), payload)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    with open_zip("01_core_project.zip") as archive:
        for path in sorted(item for item in ROOT.rglob("*") if item.is_file() and core_file(item)):
            add_path(archive, path, path.relative_to(ROOT).as_posix())

    with open_zip("02_raw_metr_la.zip") as archive:
        raw = ROOT / "data/raw/metr-la.h5"
        add_path(archive, raw, "data/raw/metr-la.h5")

    interim = ROOT / "data/interim/metr_la_eda_long.parquet"
    write_chunk_package("03_interim_data_part1.zip", interim, "metr_la_eda_long.parquet.part001", 0)
    write_chunk_package("04_interim_data_part2.zip", interim, "metr_la_eda_long.parquet.part002", CHUNK_SIZE)

    forest = ROOT / "artifacts/models/random_forest.joblib"
    write_chunk_package("05_random_forest_part1.zip", forest, "random_forest.joblib.part001", 0)
    write_chunk_package("06_random_forest_part2.zip", forest, "random_forest.joblib.part002", CHUNK_SIZE)

    for package in sorted(OUT.glob("*.zip")):
        size_mib = package.stat().st_size / 1024 / 1024
        if size_mib >= 20:
            raise ValueError(f"{package.name} 超过20MiB：{size_mib:.2f}")
        print(f"{package.name}: {size_mib:.2f} MiB")


if __name__ == "__main__":
    main()
