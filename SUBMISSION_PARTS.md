# 20MB限制分包使用说明

请上传 `dist/split/` 中全部6个ZIP文件。每个文件均小于20MB：

1. `01_core_project.zip`：Notebook、代码、报告、Streamlit Dashboard、紧凑回放数据和运行所需模型；
2. `02_raw_metr_la.zip`：METR-LA原始数据；
3. `03_interim_data_part1.zip`：清洗后EDA数据上半部分；
4. `04_interim_data_part2.zip`：清洗后EDA数据下半部分；
5. `05_random_forest_part1.zip`：随机森林模型上半部分；
6. `06_random_forest_part2.zip`：随机森林模型下半部分。

## 直接运行课程展示

只下载并解压 `01_core_project.zip`，即可阅读Notebook并运行完整Streamlit展示：

```bash
python run_dashboard.py
```

脚本会自动识别项目根目录、检查核心数据、寻找附近分包、尽可能恢复完整项目、安装两份依赖并启动Streamlit。

其余5个包用于保留原始数据、清洗后完整数据和随机森林模型，不是运行课程展示的必要条件。

## 自动恢复完整项目数据

1. 只解压 `01_core_project.zip`；
2. 将其余5个ZIP放在解压后的项目目录内，或者放在项目目录的上一级；
3. 在项目目录执行：

```bash
python run_dashboard.py
```

启动脚本会自动解压所有找到的分包，并恢复：

- `data/interim/metr_la_eda_long.parquet`
- `artifacts/models/random_forest.joblib`

恢复完成后会自动校验SHA-256并显示恢复程度，随后安装依赖并启动Streamlit。只想合并而暂不启动时执行：

```bash
python run_dashboard.py --restore-only
```

课程报告Notebook位于：

```text
notebooks/traffic_congestion_course_report.ipynb
```

仅阅读Notebook与运行Streamlit Dashboard时，不需要恢复两个大文件；重新生成EDA数据或直接加载随机森林模型时再运行恢复脚本。
