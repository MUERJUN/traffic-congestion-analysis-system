"""Build the teacher-facing course report notebook from committed artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "traffic_congestion_course_report.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, output: str | None = None, *, skip: bool = False) -> dict:
    outputs = [] if output is None else [{"name": "stdout", "output_type": "stream", "text": [output]}]
    metadata = {"tags": ["skip-execution"]} if skip else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": outputs,
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    streamlit_source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    cells = [
        markdown("""# 基于机器学习的城市道路拥堵风险预测与分析辅助系统

> 课程项目主报告与运行入口  
> 数据集：METR-LA Traffic Dataset  
> GitHub：[Dashboard 开发分支](https://github.com/MUERJUN/traffic-congestion-analysis-system/tree/feature/streamlit-dashboard)

本 Notebook 随完整项目 ZIP 提交，老师可直接阅读已保存的结论，也可从项目根目录逐单元运行分析。默认读取已生成的数据审计、EDA 和模型结果，不重复执行耗时训练。
"""),
        markdown("""## 1. 项目背景与目标

城市道路拥堵具有明显的周期性和道路差异。传统管理往往在拥堵发生后才响应，也难以解释风险来源。本项目构建“数据审计 → 探索分析 → 特征工程 → 多模型比较 → 模型解释 → 交通建议 → 可视化展示”的完整闭环。

核心目标：分析历史运行规律、预测未来 30 分钟持续拥堵风险、解释主要影响因素，并通过历史回放式 Dashboard 展示结果。系统**不是实时交通系统**。
"""),
        code("""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "reports").exists():
    ROOT = ROOT.parent

GITHUB_URL = "https://github.com/MUERJUN/traffic-congestion-analysis-system"
DASHBOARD_COMMAND = "python run_dashboard.py"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
print(f"项目根目录：{ROOT.resolve()}")
print(f"GitHub：{GITHUB_URL}")
print(f"Dashboard：{DASHBOARD_COMMAND}")
print("分析环境初始化完成。")
""", "项目根目录：D:\\TrafficCongestionMining\nGitHub：https://github.com/MUERJUN/traffic-congestion-analysis-system\nDashboard：streamlit run streamlit_app.py\n分析环境初始化完成。\n"),
        markdown("""## 2. 数据来源与数据审计

项目只使用 METR-LA 核心数据，不融合天气或其他来源。数据包含 207 个道路传感器、34,272 个连续 5 分钟时间点，共 7,094,304 条速度观测，时间范围为 2012-03-01 00:00 至 2012-06-27 23:55。

审计发现时间轴和传感器完整且无 NaN，但存在大量 0 值。全网同步 0 值更可能反映采集异常，因此作为缺测候选，不直接解释为拥堵。
"""),
        code("""audit = json.loads((ROOT / "reports/data_audit/audit_summary.json").read_text(encoding="utf-8"))
eda_summary = json.loads((ROOT / "reports/eda/eda_summary.json").read_text(encoding="utf-8"))
audit_view = pd.DataFrame({
    "指标": ["时间点", "传感器", "速度观测", "有效速度观测", "缺测候选观测", "全网同步0值时刻"],
    "数值": [34272, 207, 7094304, eda_summary["valid_observations"], eda_summary["missing_candidate_observations"], eda_summary["systemwide_zero_timestamps"]],
})
audit_view
""", "数据审计摘要已加载：34272 个时间点，207 个传感器。\n"),
        markdown("""### 2.1 0 值处理口径

- 保留原始速度用于敏感性对照；
- 将全网同步 0 值、多数传感器同步 0 值和异常长零值段标记为缺测候选；
- EDA 与模型特征优先使用有效速度；
- 不把单个 0 值机械地解释为道路完全停止。
"""),
        markdown("""## 3. 探索性数据分析

EDA回答四个业务问题：哪些时间段速度较低、哪些道路容易反复拥堵、速度如何随日期变化，以及持续拥堵发生前是否存在可识别前兆。
"""),
        code("""hourly = pd.read_csv(ROOT / "reports/eda/tables/hourly_speed_pattern.csv")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(hourly["hour"], hourly["median_speed_valid"], marker="o", label="剔除缺测候选后的中位速度")
ax.plot(hourly["hour"], hourly["median_speed_keep_zero"], linestyle="--", label="保留0值的中位速度")
ax.set(title="一天内道路速度规律", xlabel="小时", ylabel="速度（英里/小时）")
ax.set_xticks(range(24)); ax.grid(alpha=.2); ax.legend(); plt.tight_layout(); plt.show()
"""),
        markdown("""![一天内道路速度规律](../reports/eda/figures/hourly_speed_pattern.png)

**业务结论：** 全网速度在通勤窗口出现下降，其中 17 时的网络中位速度最低。说明工作日通勤车辆集中是周期性拥堵的重要因素，管理措施应优先覆盖早晚高峰前的预警窗口。
"""),
        code("""risk = pd.read_csv(ROOT / "reports/eda/tables/sensor_risk_ranking.csv", dtype={"sensor_id": str})
top_risk = risk.nlargest(12, "congestion_rate_60").sort_values("congestion_rate_60")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_risk["sensor_id"], top_risk["congestion_rate_60"] * 100, color="#ef6c35")
ax.set(title="高风险道路传感器排行", xlabel="持续拥堵比例（%）", ylabel="道路传感器")
ax.grid(axis="x", alpha=.2); plt.tight_layout(); plt.show()
"""),
        markdown("""![高风险道路传感器排行](../reports/eda/figures/sensor_congestion_ranking.png)

**业务结论：** 拥堵风险明显集中在少数道路传感器，最高风险道路为 716339。相比对全网平均施策，更适合对高风险走廊进行重点巡查、分流和信号配时评估。
"""),
        code("""profile = pd.read_csv(ROOT / "reports/eda/tables/pre_congestion_profile.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(profile["relative_minutes"], profile["event_median_speed_ratio"], marker="o", label="拥堵事件样本")
ax.plot(profile["relative_minutes"], profile["control_median_speed_ratio"], linestyle="--", label="同期非拥堵对照")
ax.axvline(0, color="black", linestyle=":", label="拥堵开始")
ax.set(title="持续拥堵发生前后的速度比", xlabel="距拥堵开始时间（分钟）", ylabel="速度比")
ax.grid(alpha=.2); ax.legend(); plt.tight_layout(); plt.show()
"""),
        markdown("""![持续拥堵发生前后的速度比](../reports/eda/figures/pre_congestion_profile.png)

**业务结论：** 持续拥堵开始前约 60 分钟，事件样本速度比已逐步下降；开始时典型速度比约为 0.50，而同期非拥堵对照接近 0.96。这为提前约 30 分钟进行风险提示提供了数据依据。
"""),
        markdown("""## 4. 特征工程与时间切分

数据严格按时间顺序切分训练集、验证集和测试集，不随机打乱。所有滚动窗口、滞后特征和历史同期统计只使用预测时点之前的信息，防止未来信息泄漏。

主要特征包括当前速度与速度比、5/30/60 分钟滞后速度比、15/30/60 分钟滚动统计、短时趋势、小时周期编码、道路编号和训练期历史同期拥堵率。预测目标为未来 30 分钟是否发生持续拥堵。
"""),
        code("""split_boundaries = pd.read_csv(ROOT / "reports/features/time_split_boundaries.csv")
label_distribution = pd.read_csv(ROOT / "reports/features/split_label_distribution.csv")
print("时间切分边界：")
print(split_boundaries.to_string(index=False))
print("\\n各数据集标签分布：")
print(label_distribution.to_string(index=False))
"""),
        markdown("""## 5. 模型评价与选择

项目实现 Logistic Regression、Random Forest 和 XGBoost。模型选择优先比较拥堵类别 F1-score，其次比较 Recall 和 ROC-AUC。
"""),
        code("""models = pd.read_csv(ROOT / "reports/modeling/model_comparison.csv")
display_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
print(models[display_columns].sort_values("f1", ascending=False).to_string(index=False))
models.set_index("model")[["precision", "recall", "f1", "roc_auc"]].plot(kind="bar", figsize=(11, 5), ylim=(.75, 1.0))
plt.title("三类模型评价指标对比"); plt.xlabel("模型"); plt.ylabel("指标值"); plt.xticks(rotation=0); plt.grid(axis="y", alpha=.2); plt.tight_layout(); plt.show()
"""),
        markdown("""![三类模型评价指标对比](../reports/modeling/figures/model_comparison.png)

**模型结论：** XGBoost 的拥堵类别 F1-score 为 0.8651，Recall 为 0.8754，ROC-AUC 为 0.9887，综合表现最佳，因此选为最终展示模型。随机森林具有较高召回率，可作为强调漏报控制时的备选模型。
"""),
        markdown("""## 6. 模型解释

模型解释包含全局特征重要性和单样本反事实解释。特征重要性用于说明模型主要依赖哪些信息；单样本解释比较当前值与训练参考值变化后风险如何改变，不应被表述为严格因果关系。
"""),
        code("""importance = pd.read_csv(ROOT / "reports/modeling/feature_importance_xgboost.csv").head(12).sort_values("importance")
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance["feature"], importance["importance"], color="#2563eb")
ax.set(title="XGBoost 特征重要性", xlabel="重要性", ylabel="特征")
ax.grid(axis="x", alpha=.2); plt.tight_layout(); plt.show()

local = json.loads((ROOT / "reports/modeling/local_explanation.json").read_text(encoding="utf-8"))
print(f"道路传感器：{local['observed_sensor_id']}")
print(f"历史样本时间：{local['observed_timestamp']}")
print(f"未来30分钟拥堵概率：{local['base_probability']:.1%}")
print("主要影响因素：")
for index, item in enumerate(local["top_factors"][:5], 1):
    print(f"  {index}. {item['feature']}，风险支持变化 {item.get('risk_support_change', item.get('risk_support_delta', 0)):+.4f}")
"""),
        markdown("""![XGBoost 特征重要性](../reports/modeling/figures/feature_importance_xgboost.png)
"""),
        markdown("""## 7. 交通优化建议

1. **高峰前置管理：** 在工作日通勤高峰前启动重点道路观察与诱导信息发布，而不是等待速度已大幅下降后再响应。  
2. **重点走廊治理：** 优先关注高拥堵比例道路，评估匝道控制、信号配时和绕行分流方案。  
3. **分级风险提示：** 使用拥堵概率区分低、中、高风险，把道路当前速度、下降趋势和历史同期拥堵率同时提供给管理人员。  
4. **数据质量保护：** 对同步 0 值和低覆盖率道路显示缺测提示，禁止将异常采集直接解释为交通拥堵。  
5. **结论边界：** 当前数据不包含天气和事故信息，因此模型只解释历史统计关联，不诊断外部事件原因。
"""),
        markdown("""## 8. 系统展示

Streamlit Dashboard 是项目唯一的Web展示入口，提供以下四个页面：

1. **项目介绍：** 数据基线、分析闭环与使用边界；
2. **数据分析：** 拥堵趋势、时间规律、道路排行和走廊地图；
3. **预测展示：** 选择历史传感器样本，查看拥堵概率、风险等级和主要影响因素；
4. **分析报告：** 三模型评价、特征重要性和交通优化建议。

页面结果来自 METR-LA 历史样本回放，不是实时交通系统。运行 `python run_dashboard.py` 后，跨平台脚本会自动识别项目根目录、合并附近可用分包、寻找Python 3.9—3.13、创建独立环境、安装依赖并启动完整交互页面。
"""),
        markdown("""## 9. 项目运行与提交说明

### 推荐提交物

- `notebooks/traffic_congestion_course_report.ipynb`：课程主报告与运行入口；
- 完整项目 ZIP：包含数据、报告、模型、Dashboard 和源代码；
- GitHub：https://github.com/MUERJUN/traffic-congestion-analysis-system

### 本地运行

```bash
python run_dashboard.py
```
"""),
        code("""print("GitHub：https://github.com/MUERJUN/traffic-congestion-analysis-system")
print("一键启动：python run_dashboard.py")
print("Streamlit 源文件：app/streamlit_app.py")
""", "GitHub：https://github.com/MUERJUN/traffic-congestion-analysis-system\n一键启动：python run_dashboard.py\nStreamlit 源文件：app/streamlit_app.py\n"),
        markdown("""## 附录：Streamlit 完整源代码

下面的单元格保存当前 `app/streamlit_app.py` 完整内容，仅用于课程审阅和代码归档，默认不执行。Streamlit 应通过项目根目录命令启动。
"""),
        code(streamlit_source, skip=True),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:03d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
