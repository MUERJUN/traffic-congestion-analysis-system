# M2 模型数据字段字典

## 1. 数据粒度

每行对应一个 `timestamp + sensor_id` 预测样本。输入特征只来自预测时点及之前60分钟；目标来自预测时点之后30分钟。

本阶段只生成可供后续模型使用的数据集，不训练模型。

## 2. 元数据与目标

| 字段 | 类型 | 用途 | 是否进入模型 |
|---|---|---|---|
| `timestamp` | datetime | 预测时点、时间顺序和复核 | 否，不直接输入；其日历派生特征进入模型 |
| `sensor_id` | string | 道路传感器类别 | 是，后续按类别编码，不可转为连续数值 |
| `split` | string | `train` / `validation` / `test` | 否 |
| `target_congestion_30m` | int8 | 未来30分钟持续拥堵标签，0或1 | 目标列 |

## 3. 标签定义

1. 每个传感器的自由流速度仅使用训练段有效速度第85百分位估计。
2. 单时点拥堵：有效速度/训练期自由流速度 < 0.60。
3. 未来30分钟对应后续6个五分钟观测。
4. 未来6点必须全部有效；否则不生成该样本。
5. 未来6点中至少3点拥堵时，`target_congestion_30m=1`，否则为0。
6. 0.50和0.70阈值只用于标签敏感性统计，不写入模型输入文件，避免误用。

## 4. 时间特征

| 字段 | 类型 | 来源时点 | 说明 |
|---|---|---|---|
| `hour` | int8 | 0 | 小时0—23 |
| `day_of_week` | int8 | 0 | 星期一为0，星期日为6 |
| `is_weekend` | int8 | 0 | 周六、周日为1 |
| `hour_sin` / `hour_cos` | float | 0 | 小时周期编码 |
| `day_of_week_sin` / `day_of_week_cos` | float | 0 | 星期周期编码 |

## 5. 当前状态与滞后特征

| 字段 | 时间偏移 | 说明 |
|---|---:|---|
| `current_speed_mph` | 0 | 当前有效速度 |
| `free_flow_speed_train_mph` | 训练期统计 | 训练期第85百分位自由流速度 |
| `current_speed_ratio` | 0 | 当前速度/训练期自由流速度 |
| `lag_speed_ratio_5m` | -5分钟 | 5分钟前速度比 |
| `lag_speed_ratio_10m` | -10分钟 | 10分钟前速度比 |
| `lag_speed_ratio_15m` | -15分钟 | 15分钟前速度比 |
| `lag_speed_ratio_30m` | -30分钟 | 30分钟前速度比 |
| `lag_speed_ratio_60m` | -60分钟 | 60分钟前速度比 |

样本要求偏移-60分钟至当前的13个五分钟观测全部有效，不对缺口插值。

## 6. 滚动统计

对15、30、60分钟三个窗口分别生成：

- `rolling_speed_ratio_mean_*`：速度比均值。
- `rolling_speed_ratio_min_*`：速度比最小值。
- `rolling_speed_ratio_std_*`：速度比总体标准差（`ddof=0`）。

所有窗口均向后滚动并包含当前时点，不包含未来观测。

## 7. 趋势特征

| 字段 | 说明 |
|---|---|
| `trend_speed_mph_per_min_10m` | 当前速度与10分钟前速度之差/10 |
| `trend_speed_mph_per_min_15m` | 当前速度与15分钟前速度之差/15 |
| `trend_speed_mph_per_min_30m` | 当前速度与30分钟前速度之差/30 |
| `speed_change_5m` | 当前速度减5分钟前速度 |
| `consecutive_decline_steps` | 截至当前连续下降的五分钟步数 |

负趋势或较大的连续下降步数表示速度正在恶化，但仍是统计特征，不是事故因果证据。

## 8. 训练期历史同期特征

历史同期分组键为：`sensor_id + 工作日/周末 + 五分钟时段`。统计只从训练段生成；验证和测试复用相同查找表。

| 字段 | 说明 |
|---|---|
| `historical_period_median_speed_mph_train` | 训练期同传感器、同日期类型、同时段速度中位数 |
| `historical_period_congestion_rate_train` | 训练期同组单时点拥堵率 |
| `deviation_from_historical_median_mph` | 当前速度与训练期历史同期中位数之差 |

若训练期某个细分组合没有有效数据，回退到该传感器训练期总体中位速度或总体拥堵率，不读取验证/测试数据补全。

## 9. 文件模式

每个Split Parquet包含：

- 3个元数据字段：`timestamp`、`sensor_id`、`split`。
- 1个目标字段：`target_congestion_30m`。
- 32个数值/布尔特征。
- `sensor_id`作为后续第33个类别模型特征。
- 合计36列。

所有三份文件均已验证：36列完整、无空值、`timestamp + sensor_id`唯一、207个传感器完整、标签域为`{0,1}`。

