# 基于机器学习的城市道路拥堵风险预测与分析辅助系统

## 项目介绍

本项目基于 METR-LA 历史道路传感器数据，建设一个道路拥堵风险预测与交通分析辅助系统。项目目标不是只训练单一模型，而是形成完整的数据分析闭环：

> 问题提出 → 数据获取 → 数据审计与清洗 → 探索性分析 → 模型建立与比较 → 模型解释 → 业务结论与建议 → Streamlit可视化展示

最终产品定位为**历史交通数据回放式分析平台**，用于课程研究和辅助分析，不是实时交通监控或控制系统。

## 核心能力规划

- 分析不同时间段的道路速度和拥堵规律。
- 识别历史上容易出现拥堵的道路传感器。
- 预测选定历史时点之后30分钟的拥堵风险。
- 展示拥堵概率、风险等级和主要影响因素。
- 比较 Logistic Regression、Random Forest、XGBoost。
- 通过 Streamlit 展示数据分析、预测和模型报告。

## 当前进度

| 阶段 | 状态 | 主要产物 |
|---|---|---|
| Phase 1：项目设计 | 已完成 | `PROJECT_DESIGN_OUTLINE.md` |
| Phase 2：数据获取与审计 | 已完成 | `reports/data_audit`、`reports/eda_preparation` |
| 开发管理基线 | 本轮已建立，待确认 | `docs`、双轨分支规划、验收标准 |
| 数据清洗与正式EDA | 已完成 | 清洗流水线、6张统计表、6张图和EDA报告 |
| 标签与历史特征 | 已完成 | 未来30分钟标签、33个模型特征来源、严格时间切分 |
| 三模型训练与比较 | 已完成 | 三模型指标、XGBoost最终模型和解释产物 |
| Streamlit Dashboard | 未开始 | 尚无页面代码 |

Track A现已包含可复现的数据清洗、正式EDA、未来30分钟标签、历史特征和时间切分代码；当前仍没有训练结果、模型文件或Streamlit页面。

## 数据来源

核心数据使用 [METR-LA Traffic Dataset](https://github.com/liyaguang/DCRNN)，不融合天气或其他交通数据集。

已审计数据基线：

- 34,272个时间点。
- 207个道路传感器。
- 五分钟采样。
- 7,094,304个速度观测。
- 实际文件时间范围为2012-03-01 00:00至2012-06-27 23:55。
- 时间轴连续，无重复和断点。
- 数据无NaN和负值，但存在大量0值。
- 全网同步0被视为系统性缺测候选，不能直接解释为真实拥堵。

详细来源、哈希、字段和质量结论见：

- [数据来源登记](reports/data_audit/DATA_SOURCE_REGISTRY.md)
- [数据字典](reports/data_audit/DATA_DICTIONARY.md)
- [数据审计报告](reports/data_audit/DATA_AUDIT_REPORT.md)
- [EDA准备方案](reports/eda_preparation/EDA_PREPARATION_PLAN.md)

原始数据保存在本地 `data/raw`，受 `.gitignore` 保护，不直接提交到Git仓库。

## 开发路线

项目采用双轨开发。

### Track A：算法与数据分析

- 计划分支：`feature/model-pipeline`
- 范围：清洗、EDA、标签、特征、模型训练、评价、解释。
- 必选模型：Logistic Regression、Random Forest、XGBoost。
- 必选指标：Accuracy、Precision、Recall、F1-score、ROC-AUC。
- 模型选择：优先拥堵类F1，其次Recall、ROC-AUC。

详细规格：[MODEL_TRACK_SPEC.md](docs/MODEL_TRACK_SPEC.md)

### Track B：Streamlit系统展示

- 计划分支：`feature/streamlit-dashboard`
- 范围：项目介绍、数据分析、预测展示、分析报告四个页面。
- 定位：历史交通数据回放，不描述为实时系统。
- 模型结果由Track A的版本化产物提供，不在页面中伪造或重新训练。

详细规格：[DASHBOARD_TRACK_SPEC.md](docs/DASHBOARD_TRACK_SPEC.md)

整体路线与轨道依赖见 [DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md)，各阶段放行条件见 [ACCEPTANCE_CRITERIA.md](docs/ACCEPTANCE_CRITERIA.md)。

## Phase 3A运行

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
traffic-phase3
traffic-m2
traffic-m3
```

主要输出：

- [数据清洗报告](reports/data_cleaning/DATA_CLEANING_REPORT.md)
- [正式EDA报告](reports/eda/EDA_REPORT.md)
- [Phase 3A验收记录](reports/PHASE3A_ACCEPTANCE_REPORT.md)
- [M2标签与特征报告](reports/features/M2_REPORT.md)
- [M2字段字典](reports/features/FEATURE_DICTIONARY.md)
- [M2验收记录](reports/M2_ACCEPTANCE_REPORT.md)
- [M3模型报告](reports/modeling/M3_REPORT.md)
- [M3验收记录](reports/M3_ACCEPTANCE_REPORT.md)
- `data/interim/metr_la_eda_long.parquet`（本地生成，受Git忽略）

## 开发原则

1. 时间序列不随机切分。
2. 禁止未来信息泄漏。
3. 第一版不使用深度学习。
4. 不增加天气数据，不随意融合异源数据。
5. 0值和全网同步0按数据质量策略处理。
6. 所有业务结论必须包含可追溯的数据依据。
7. 模型解释代表统计关联和模型依据，不冒充交通因果诊断。

## 仓库结构（当前）

```text
.
├── data/                         # 本地数据；原始/处理数据默认不提交
├── config/                       # 可复现运行配置
├── docs/                         # 开发计划、双轨规格和验收标准
├── reports/
│   ├── data_audit/               # 数据来源、字典、审计报告与质量表
│   ├── data_cleaning/            # Phase 3A清洗报告与统计
│   ├── eda/                      # 正式EDA报告、表格和图表
│   └── eda_preparation/          # EDA准备方案
├── src/traffic_congestion/       # Track A清洗与EDA实现
├── tests/                        # 自动测试
├── pyproject.toml                # Python项目与依赖
├── PROJECT_DESIGN_OUTLINE.md     # 已确认的项目设计基线
├── PROJECT_STATUS.md             # 当前阶段状态
└── README.md
```

## 当前开发状态说明

当前位于Track A的`feature/model-pipeline`开发阶段，Phase 3A、M2和M3已完成。最终模型为XGBoost；下一步是Track B Streamlit页面与系统集成。
