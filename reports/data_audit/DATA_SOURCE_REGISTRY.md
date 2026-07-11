# METR-LA 数据来源登记

## 1. 核心速度数据

| 项目 | 登记内容 |
|---|---|
| 文件 | `data/raw/metr-la.h5` |
| 用途 | METR-LA 五分钟道路传感器速度矩阵 |
| 作者指向来源 | DCRNN 作者仓库 README 中的 Google Drive 文件 ID `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX` |
| 作者仓库 | <https://github.com/liyaguang/DCRNN> |
| 本次实际下载 | Hugging Face 公开镜像 `jimmygao3218/METRLA`，经 `hf-mirror.com` 解析下载 |
| 镜像页面 | <https://huggingface.co/datasets/jimmygao3218/METRLA/blob/main/metr-la.h5> |
| 下载时间 | 2026-07-11（Asia/Shanghai） |
| 文件大小 | 57,038,056 bytes |
| SHA-256 | `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05` |

### 来源说明

作者仓库提供的 Google Drive 直链在当前网络环境下连续连接超时，未产生文件。因此使用公开镜像完成获取。镜像页面公开的文件大小与 SHA-256 和本地文件完全一致；本地 HDF5 的 34,272×207 结构、五分钟时间轴及传感器编号顺序也与作者仓库说明和同源元数据一致。

但作者仓库没有公布原始 Google Drive 文件的校验和，因此目前能确认的是“镜像文件内部结构与公开基准一致”，不能声称已通过作者官方哈希证明逐字节同一。若后续获得作者直链文件，应再次计算哈希并比较。

## 2. 同源传感器元数据

下列文件直接下载自 DCRNN 作者 GitHub 仓库 `data/sensor_graph`：

| 文件 | 大小（bytes） | SHA-256 |
|---|---:|---|
| `data/metadata/graph_sensor_ids.txt` | 1,448 | `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990` |
| `data/metadata/graph_sensor_locations.csv` | 6,341 | `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f` |
| `data/metadata/distances_la_2012.csv` | 6,393,348 | `a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5` |
| `data/metadata/adj_mx.pkl` | 680,459 | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |

这些元数据与核心数据同属 METR-LA/DCRNN 数据体系，不属于新增异源天气或第三方业务数据。

## 3. 完整校验清单

机器可读校验记录见 `file_checksums.csv`。原始 HDF5 和元数据文件不得直接修改；后续转换结果必须写入 `data/interim` 或 `data/processed`。

