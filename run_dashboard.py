"""Install project dependencies and launch the Streamlit dashboard."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parent
OPTIONAL_PACKAGE_NAMES = {
    "02_raw_metr_la.zip",
    "03_interim_data_part1.zip",
    "04_interim_data_part2.zip",
    "05_random_forest_part1.zip",
    "06_random_forest_part2.zip",
}
RESTORABLE_FILES = {
    "data/interim/metr_la_eda_long.parquet": (
        ("metr_la_eda_long.parquet.part001", "metr_la_eda_long.parquet.part002"),
        "316aaa550a7c3d66d0f7032653b6f85111463bb5e8f8a50cc186824bcdc952ec",
    ),
    "artifacts/models/random_forest.joblib": (
        ("random_forest.joblib.part001", "random_forest.joblib.part002"),
        "7280eb50a699566b8af4198798ab4c2bc2112ae13f81209a31c3f8f679de0c32",
    ),
}
REQUIRED_FILES = (
    "requirements.txt",
    "requirements-dashboard.txt",
    "streamlit_app.py",
    "app/streamlit_app.py",
    "data/dashboard/test_replay.parquet",
    "data/dashboard/congestion_heatmap.parquet",
    "data/dashboard/train_reference.parquet",
    "artifacts/models/xgboost.joblib",
)


def verify_project(root: Path = ROOT) -> None:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"项目文件不完整，缺少：\n{formatted}")
    print("[1/3] 项目文件检查通过。")


def discover_optional_packages(root: Path = ROOT) -> list[Path]:
    """Find optional submission ZIPs beside or inside the extracted core project."""
    search_dirs = (root, root.parent, root / "dist" / "split")
    found: dict[str, Path] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for package in directory.glob("*.zip"):
            if package.name in OPTIONAL_PACKAGE_NAMES:
                found.setdefault(package.name, package.resolve())
    return [found[name] for name in sorted(found)]


def safe_extract_package(package: Path, root: Path = ROOT) -> None:
    """Extract a known package while preventing paths from escaping the project root."""
    root_resolved = root.resolve()
    with ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise BadZipFile(f"压缩包损坏：{package.name}")
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if destination != root_resolved and root_resolved not in destination.parents:
                raise ValueError(f"压缩包包含不安全路径：{member.filename}")
        archive.extractall(root)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def restore_file_from_parts(root: Path, relative_target: str, part_names: tuple[str, ...], expected_hash: str) -> bool:
    parts = [root / "submission_parts" / name for name in part_names]
    if not all(part.exists() for part in parts):
        return False
    target = root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    if file_sha256(target) != expected_hash:
        target.unlink(missing_ok=True)
        raise ValueError(f"{relative_target} 分片校验失败")
    print(f"  已恢复并校验：{relative_target}")
    return True


def restore_maximum_available(root: Path = ROOT) -> dict[str, bool]:
    """Extract every available optional package and restore all complete file pairs."""
    packages = discover_optional_packages(root)
    if packages:
        print(f"发现 {len(packages)} 个可选分包，正在自动合并：")
    else:
        print("未发现可选分包，使用核心包运行。")
    for package in packages:
        print(f"  解压：{package.name}")
        safe_extract_package(package, root)

    restored = {}
    for relative_target, (part_names, expected_hash) in RESTORABLE_FILES.items():
        target = root / relative_target
        if target.exists() and file_sha256(target) == expected_hash:
            restored[relative_target] = True
            continue
        restored[relative_target] = restore_file_from_parts(root, relative_target, part_names, expected_hash)

    status = {
        "raw_data": (root / "data/raw/metr-la.h5").exists(),
        "interim_data": restored["data/interim/metr_la_eda_long.parquet"],
        "random_forest": restored["artifacts/models/random_forest.joblib"],
    }
    completed = sum(status.values())
    print(f"恢复程度：核心展示完整，可选资源 {completed}/3。")
    for label, ready in status.items():
        print(f"  {'[已有]' if ready else '[缺少]'} {label}")
    return status


def install_requirements(root: Path = ROOT) -> None:
    for index, filename in enumerate(("requirements.txt", "requirements-dashboard.txt"), start=1):
        print(f"[2/3] 正在安装依赖（{index}/2）：{filename}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(root / filename),
            ],
            cwd=root,
        )


def launch_streamlit(port: int, root: Path = ROOT) -> int:
    print(f"[3/3] 正在启动系统：http://localhost:{port}")
    print("按 Ctrl+C 可以停止系统。")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(root / "streamlit_app.py"),
            "--server.port",
            str(port),
        ],
        cwd=root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键安装依赖并启动交通拥堵分析系统")
    parser.add_argument("--check", action="store_true", help="只检查核心文件，不安装或启动")
    parser.add_argument("--restore-only", action="store_true", help="只自动合并可用分包，不安装或启动")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装，直接启动")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit端口，默认8501")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_project()
        restore_maximum_available()
        if args.check:
            print("核心包可以运行Streamlit Dashboard。")
            return 0
        if args.restore_only:
            print("可用分包合并完成。")
            return 0
        if not args.skip_install:
            install_requirements()
        return launch_streamlit(args.port)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
