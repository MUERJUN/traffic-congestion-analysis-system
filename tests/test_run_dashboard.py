from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from run_dashboard import discover_optional_packages, safe_extract_package


def test_discovers_optional_packages_next_to_extracted_core(tmp_path: Path) -> None:
    root = tmp_path / "core"
    root.mkdir()
    expected = tmp_path / "02_raw_metr_la.zip"
    expected.write_bytes(b"placeholder")
    (tmp_path / "unrelated.zip").write_bytes(b"placeholder")

    packages = discover_optional_packages(root)

    assert packages == [expected]


def test_safely_extracts_package_into_project_root(tmp_path: Path) -> None:
    root = tmp_path / "core"
    root.mkdir()
    package = tmp_path / "03_interim_data_part1.zip"
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        archive.writestr("submission_parts/sample.part001", b"traffic-data")

    safe_extract_package(package, root)

    assert (root / "submission_parts/sample.part001").read_bytes() == b"traffic-data"


def test_rejects_zip_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "core"
    root.mkdir()
    package = tmp_path / "04_interim_data_part2.zip"
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        archive.writestr("../outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="不安全"):
        safe_extract_package(package, root)


def test_launcher_source_uses_gbk_safe_status_text() -> None:
    source = (Path(__file__).resolve().parents[1] / "run_dashboard.py").read_text(encoding="utf-8")

    assert "✓" not in source
    assert "[已有]" in source
