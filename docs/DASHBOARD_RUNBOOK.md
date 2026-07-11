# Dashboard 运行说明

Dashboard 位于 `feature/streamlit-dashboard` 分支，定位为 METR-LA 历史数据回放平台，不是实时交通系统。

## 启动

```bash
python -m pip install -r requirements-dashboard.txt
streamlit run streamlit_app.py
```

当前提供四页面骨架、统一视觉规范、非实时声明、风险阈值规则，以及 Track A 产物缺失或不兼容时的安全空状态。页面不会生成或展示虚构模型结果。

## Track A 对接

Dashboard 查找 `artifacts/eda/dashboard_summary.json`、`artifacts/predictions/replay_cases.json` 和 `artifacts/metrics/model_comparison.json`。JSON 必须声明 `schema_version: "1.0"` 并包含加载器要求的字段。
