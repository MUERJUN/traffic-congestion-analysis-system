"""Version-aware access to Track A artifacts; no model output is calculated here."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = "1.0"

@dataclass(frozen=True)
class ArtifactResult:
    data: dict[str, Any] | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.data is not None and self.error is None

def load_json_artifact(path: Path, required_fields: set[str]) -> ArtifactResult:
    if not path.exists():
        return ArtifactResult(None, f"等待 Track A 产物：{path.name} 尚不存在。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ArtifactResult(None, f"无法读取 {path.name}：{exc}")
    if not isinstance(payload, dict):
        return ArtifactResult(None, f"{path.name} 的根节点必须是 JSON 对象。")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        return ArtifactResult(None, f"{path.name} 模式版本不兼容：需要 {SUPPORTED_SCHEMA_VERSION}。")
    missing = sorted(required_fields - payload.keys())
    if missing:
        return ArtifactResult(None, f"{path.name} 缺少必要字段：{', '.join(missing)}。")
    return ArtifactResult(payload)
