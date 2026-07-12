import json
from src.app.artifacts import SUPPORTED_SCHEMA_VERSION, load_json_artifact

def test_missing_artifact(tmp_path):
    result = load_json_artifact(tmp_path / "missing.json", {"schema_version"})
    assert not result.available and "尚不存在" in result.error

def test_rejects_incompatible_schema(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"schema_version": "0.9", "items": []}), encoding="utf-8")
    assert "模式版本不兼容" in load_json_artifact(path, {"schema_version", "items"}).error

def test_rejects_missing_fields(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"schema_version": SUPPORTED_SCHEMA_VERSION}), encoding="utf-8")
    assert "items" in load_json_artifact(path, {"schema_version", "items"}).error

def test_accepts_compatible_artifact(tmp_path):
    payload = {"schema_version": SUPPORTED_SCHEMA_VERSION, "items": []}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_json_artifact(path, set(payload)).data == payload
