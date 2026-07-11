# M4 Streamlit Dashboard 验收记录

## 结论

Track B四页面历史回放Dashboard已实现并通过本地启动与HTTP烟测，可进入端到端集成验收。

## 页面验收

- [x] 项目介绍页面：展示项目目标、数据基线、分析闭环和使用边界。
- [x] 数据分析页面：展示时间规律、道路风险排行、速度趋势和拥堵前兆。
- [x] 预测展示页面：选择传感器和历史时点，输出概率、风险等级、当前状态和主要因素。
- [x] 分析报告页面：展示模型指标、ROC图、特征重要性和业务建议。
- [x] 四页面均显示“历史交通数据回放，非实时交通系统”。

## 产物对接

- [x] EDA页面读取`reports/eda/tables`和图表，不重复计算全量EDA。
- [x] 预测页面读取固定`artifacts/models/xgboost.joblib`。
- [x] 预测阈值读取M3 Validation固定阈值0.85，不在页面修改。
- [x] 报告页读取`model_comparison.csv`和XGBoost重要性文件。
- [x] 模型或数据缺失时显示明确空状态，不伪造概率或指标。
- [x] 历史样本的事后标签单独展示，不作为模型输入。

## 自动验证

- Streamlit健康检查`/_stcore/health`返回200。
- Streamlit首页HTTP检查返回200。
- Python静态编译通过。
- 全部单元测试通过。
- 风险等级0.40/0.70边界测试通过。

## 运行方式

```powershell
python -m pip install -e .
streamlit run app/streamlit_app.py
```

模型二进制和M2 Parquet属于本地大文件，受`.gitignore`保护；运行本地Dashboard前需确保Track A产物存在。

