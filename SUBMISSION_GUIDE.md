# 课程项目提交说明

## 推荐提交内容

1. 主报告 Notebook：`notebooks/traffic_congestion_course_report.ipynb`
2. 完整项目压缩包：`dist/traffic-congestion-analysis-system-course-submission.zip`
3. GitHub Dashboard 分支：<https://github.com/MUERJUN/traffic-congestion-analysis-system/tree/feature/streamlit-dashboard>

Notebook 是课程报告和分析运行入口，包含数据审计、EDA、时间切分、模型比较、模型解释、业务建议、Streamlit 页面说明以及完整 Streamlit 源代码附录。模型默认读取已有结果，不重复训练。

为控制课程平台上传体积，提交 ZIP 不包含三个可重新生成的模型特征矩阵：`train_features.parquet`、`validation_features.parquet` 和 `test_features.parquet`。ZIP 仍保留 METR-LA 原始数据、清洗后 EDA 数据、训练完成的模型、全部报告和可直接运行的 Dashboard；需要重新训练时可按项目脚本重新生成特征矩阵。

20MB分包方案中的 `01_core_project.zip` 已包含专供Streamlit使用的紧凑回放数据、热力图聚合数据和训练参考数据，因此核心包单独解压后即可运行四个页面，不依赖三个大型特征矩阵。

## 环境安装

在项目根目录执行：

```bash
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

## 运行 Notebook

使用 Jupyter 打开：

```text
notebooks/traffic_congestion_course_report.ipynb
```

Notebook 应从项目根目录启动，以便正确读取 `reports/`、`data/` 和 `artifacts/`。

## 运行 Streamlit Dashboard

```bash
python run_dashboard.py
```

脚本跨Windows、macOS和Linux运行，会自动识别项目根目录，在当前目录和上一级寻找其余分包，尽可能解压并恢复原始数据、清洗数据和随机森林模型。随后自动寻找Python 3.9—3.13、创建 `.runtime-venv` 项目环境、安装 `requirements.txt` 与 `requirements-dashboard.txt`，最后启动Streamlit。已安装依赖时可执行 `python run_dashboard.py --skip-install`。

如果当前命令使用的是旧版Python，脚本本身仍可解析并会自动寻找系统中的兼容解释器；电脑没有任何Python 3.9以上版本时，会显示明确的升级提示。

## 项目边界

- 页面展示的是 METR-LA 历史交通数据回放，不是实时交通系统。
- 数据未融合天气或事故来源。
- 全网同步 0 值按缺测候选处理，不直接解释为拥堵。
- 模型解释表示历史统计关联和反事实风险变化，不代表严格因果关系。
