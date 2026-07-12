"""Restore large files after all six course-submission ZIPs are extracted."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "submission_parts"
FILES = {
    ROOT / "data/interim/metr_la_eda_long.parquet": (
        [PARTS / "metr_la_eda_long.parquet.part001", PARTS / "metr_la_eda_long.parquet.part002"],
        "316aaa550a7c3d66d0f7032653b6f85111463bb5e8f8a50cc186824bcdc952ec",
    ),
    ROOT / "artifacts/models/random_forest.joblib": (
        [PARTS / "random_forest.joblib.part001", PARTS / "random_forest.joblib.part002"],
        "7280eb50a699566b8af4198798ab4c2bc2112ae13f81209a31c3f8f679de0c32",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for target, (parts, expected_hash) in FILES.items():
        missing = [part for part in parts if not part.exists()]
        if missing:
            raise FileNotFoundError(f"缺少分片：{', '.join(str(item) for item in missing)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
        actual_hash = sha256(target)
        if actual_hash != expected_hash:
            target.unlink(missing_ok=True)
            raise ValueError(f"{target.name} 校验失败，请重新解压对应分包")
        print(f"已恢复并校验：{target.relative_to(ROOT)}")
    print("完整项目文件恢复完成。")


if __name__ == "__main__":
    main()
