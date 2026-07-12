import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "traffic_congestion_course_report.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_course_notebook_contains_complete_report_sections():
    notebook = load_notebook()
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    for heading in [
        "项目背景与目标",
        "数据来源与数据审计",
        "探索性数据分析",
        "特征工程与时间切分",
        "模型评价与选择",
        "模型解释",
        "交通优化建议",
        "系统展示",
        "项目运行与提交说明",
        "Streamlit 完整源代码",
    ]:
        assert heading in markdown


def test_course_notebook_is_project_entry_without_importing_internal_modules():
    notebook = load_notebook()
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "skip-execution" not in cell.get("metadata", {}).get("tags", [])
    )
    appendix_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "skip-execution" in cell.get("metadata", {}).get("tags", [])
    )
    assert "from src" not in code
    assert "from app" not in code
    assert "github.com/MUERJUN/traffic-congestion-analysis-system" in code
    assert "python run_dashboard.py" in code
    assert "app/streamlit_app.py" in code
    assert "def main()" in appendix_code


def test_course_notebook_has_saved_outputs_and_no_errors():
    notebook = load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert any(cell.get("outputs") for cell in code_cells)
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_course_notebook_has_valid_unique_cell_ids():
    notebook = load_notebook()
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert all(isinstance(cell_id, str) and 1 <= len(cell_id) <= 64 for cell_id in cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def test_split_submission_zips_are_windows_explorer_compatible():
    packages = sorted((ROOT / "dist" / "split").glob("*.zip"))
    all_names = set()
    for package in packages:
        with zipfile.ZipFile(package) as archive:
            entries = archive.infolist()
            all_names.update(entry.filename for entry in entries)
            assert archive.testzip() is None
            assert all(not entry.filename.startswith("./") for entry in entries)
            assert all(entry.create_system == 0 for entry in entries)
            assert all(entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED} for entry in entries)

    assert "notebooks/traffic_congestion_course_report.ipynb" in all_names
    assert "data/raw/metr-la.h5" in all_names


def test_split_submission_packages_are_below_twenty_megabytes():
    packages = sorted((ROOT / "dist" / "split").glob("*.zip"))

    assert len(packages) == 6
    assert all(package.stat().st_size < 20 * 1024 * 1024 for package in packages)
    assert {package.name for package in packages} == {
        "01_core_project.zip",
        "02_raw_metr_la.zip",
        "03_interim_data_part1.zip",
        "04_interim_data_part2.zip",
        "05_random_forest_part1.zip",
        "06_random_forest_part2.zip",
    }


def test_submission_uses_streamlit_without_html_dashboard():
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")

    assert not (ROOT / "html_dashboard").exists()
    assert "html_dashboard" not in notebook_text
    assert "HTML Dashboard" not in notebook_text
    assert "python run_dashboard.py" in notebook_text


def test_one_click_dashboard_launcher_is_self_contained():
    launcher = (ROOT / "run_dashboard.py").read_text(encoding="utf-8")

    assert "requirements.txt" in launcher
    assert "requirements-dashboard.txt" in launcher
    assert "sys.executable" in launcher
    assert '"streamlit"' in launcher
    assert '"run"' in launcher
    assert "--check" in launcher
