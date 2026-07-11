# Dashboard 运行说明

Dashboard 位于 `feature/streamlit-dashboard` 分支，定位为 METR-LA 历史数据回放平台，不是实时交通系统。

## 启动

```bash
python -m pip install -r requirements-dashboard.txt
streamlit run streamlit_app.py
```

当前提供四个可交互页面、统一视觉规范、非实时声明、风险阈值规则，以及Track A产物缺失时的安全空状态。页面不会重新训练或展示虚构模型结果。

## Track A 对接

Dashboard读取：`reports/eda/tables`、`reports/eda/figures`、`reports/modeling/model_comparison.csv`、`reports/modeling/feature_importance_xgboost.csv`、`reports/modeling/m3_summary.json`、`data/processed/model_dataset/test_features.parquet`和`artifacts/models/xgboost.joblib`。缺失产物时显示明确空状态。
