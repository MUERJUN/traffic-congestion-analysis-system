# Course Submission Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a readable, runnable main Notebook and a clean project ZIP for course submission.

**Architecture:** The Notebook is the report and entry point inside the complete project, so it reads existing METR-LA audit, EDA, model, and explanation artifacts without retraining by default. Dashboard screenshots are stored under `reports/dashboard/`, while the ZIP contains the full repository content required to reproduce the analysis and launch the dashboard.

**Tech Stack:** Jupyter Notebook JSON, Python, Pandas, Matplotlib, Plotly, Streamlit, PowerShell ZIP packaging.

## Global Constraints

- The Notebook must run from the repository root and may depend on files included in the project ZIP.
- Default execution must not retrain Logistic Regression, Random Forest, or XGBoost.
- All explanatory text, chart titles, and business conclusions must be Chinese.
- Streamlit pages and launch commands must be documented in the Notebook.
- GitHub URL and HTML/Streamlit launch commands must appear near the beginning and end.
- ZIP packaging must exclude `.git`, caches, temporary test directories, and any previously generated ZIP.

---

### Task 1: Submission contract test

**Files:**
- Create: `tests/test_course_notebook.py`
- Create: `notebooks/traffic_congestion_course_report.ipynb`

**Interfaces:**
- Consumes: Notebook JSON
- Produces: assertions for required sections, code independence from `src`/`app`, Streamlit page documentation, GitHub URL, and launch commands

- [ ] Write a failing pytest that loads the Notebook as JSON and checks all required report sections and source-code appendix.
- [ ] Run the focused test and verify it fails because the Notebook does not exist.
- [ ] Create the Notebook with report cells, executable analysis cells, Streamlit page documentation, and source-code appendix.
- [ ] Run the focused test and verify it passes.

### Task 2: Streamlit presentation and execution

**Files:**
- Modify: `notebooks/traffic_congestion_course_report.ipynb`

**Interfaces:**
- Consumes: local Streamlit dashboard at port 8501
- Produces: Streamlit page documentation and an executed Notebook with saved chart outputs

- [ ] Document the four Streamlit pages and the launch command.
- [ ] Execute the Notebook from the repository root with a temporary Jupyter runtime.
- [ ] Verify every code cell finishes without an error output.

### Task 3: Submission documentation and ZIP

**Files:**
- Create: `SUBMISSION_GUIDE.md`
- Create: `dist/traffic-congestion-analysis-system-course-submission.zip`

**Interfaces:**
- Consumes: complete project and main Notebook
- Produces: teacher-facing submission instructions and uploadable project archive

- [ ] Document the Notebook, GitHub URL, HTML preview command, Streamlit command, and environment installation command.
- [ ] Package the project while excluding version-control metadata, caches, temporary files, and `dist` itself.
- [ ] Inspect ZIP contents and verify the Notebook, source, reports, dashboard, requirements, and README are included.
- [ ] Run the full pytest suite and report artifact sizes and paths.
