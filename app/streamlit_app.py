"""Historical METR-LA congestion analysis and risk replay dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.dashboard_utils import (  # noqa: E402
    ROOT as PROJECT_ROOT,
    explain_history_row,
    load_csv,
    load_congestion_heatmap,
    load_image_bytes,
    load_json,
    load_model,
    load_sensor_edges,
    load_sensor_locations,
    load_sensor_history,
    localize_feature_column,
    predict_history_row,
    risk_level,
)
from src.app.styles import CSS as SHARED_DASHBOARD_CSS  # noqa: E402


st.set_page_config(
    page_title="城市道路拥堵风险分析",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(SHARED_DASHBOARD_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .metric-note { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_replay_timestamp(value: object) -> str:
    """Use an unambiguous 24-hour timestamp in the replay controls."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M")


def format_factor_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("当前值", "训练参考值"):
        if column in result:
            result[column] = result[column].map(
                lambda value: f"{float(value):.3f}" if pd.notna(value) else "-"
            )
    if "风险支持变化" in result:
        result["风险支持变化"] = result["风险支持变化"].map(
            lambda value: f"{float(value):+.4f}" if pd.notna(value) else "-"
        )
        result["影响方向"] = result["风险支持变化"].map(
            lambda value: "提高拥堵风险" if float(value) > 0 else "降低拥堵风险"
        )
    return result


def format_model_metrics(frame: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "最终选择", result["model"].eq(selected_model).map({True: "✓", False: ""}))
    result["model"] = result["model"].replace(
        {"Logistic Regression": "逻辑回归", "Random Forest": "随机森林", "XGBoost": "XGBoost"}
    )
    rename = {
        "model": "模型",
        "accuracy": "准确率",
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1分数",
        "roc_auc": "ROC-AUC",
        "threshold": "分类阈值",
        "validation_f1": "验证集F1",
        "validation_recall": "验证集召回率",
        "validation_roc_auc": "验证集ROC-AUC",
    }
    result = result.rename(columns=rename)
    for column in result.columns:
        if column not in ("最终选择", "模型"):
            result[column] = result[column].map(
                lambda value: f"{float(value):.4f}" if pd.notna(value) else "-"
            )
    return result


def risk_badge(level: str) -> None:
    colors = {"高风险": "#ef4444", "中风险": "#f59e0b", "低风险": "#22c55e"}
    color = colors.get(level, "#64748b")
    st.markdown(
        f'<div style="display:inline-block;padding:.35rem .75rem;border-radius:999px;'
        f'background:{color};color:white;font-weight:700">风险等级：{level}</div>',
        unsafe_allow_html=True,
    )


def show_replay_banner() -> None:
    st.markdown(
        '<div class="replay-banner">历史交通数据回放平台：页面结果来自METR-LA历史样本，不是实时交通系统。</div>',
        unsafe_allow_html=True,
    )


def page_intro() -> None:
    st.markdown(
        """
        <section class="hero-panel">
          <span class="eyebrow">METR-LA 历史交通数据回放</span>
          <h1>城市道路拥堵风险预测与分析辅助系统</h1>
          <p>
            面向课程项目的交通数据挖掘 Demo：从历史速度数据出发，完成数据审计、规律分析、
            拥堵风险预测、模型解释和交通优化建议展示。页面结果来自历史样本，不是实时交通系统。
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="kpi-grid">
          <div class="kpi-card"><span>道路监测点</span><strong>207</strong></div>
          <div class="kpi-card"><span>采样间隔</span><strong>5分钟</strong></div>
          <div class="kpi-card"><span>历史时间范围</span><strong>2012.03-06</strong></div>
          <div class="kpi-card"><span>建模目标</span><strong>未来30分钟</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dashboard-grid">
          <div class="dashboard-card">
            <h3>项目闭环</h3>
            <div class="process-bar">
              <span>问题提出</span><span>数据审计</span><span>EDA</span><span>特征工程</span>
              <span>模型比较</span><span>模型解释</span><span>业务建议</span>
            </div>
          </div>
          <div class="dashboard-card">
            <h3>系统能力</h3>
            <ul>
              <li>查看道路速度趋势、拥堵时间规律和风险排行</li>
              <li>按日期和小时回放历史道路走廊拥堵分布</li>
              <li>选择传感器编号，展示拥堵概率、风险等级和主要因素</li>
              <li>对比三类机器学习模型并输出优化建议</li>
            </ul>
          </div>
          <div class="dashboard-card">
            <h3>数据基线</h3>
            <ul>
              <li>核心数据：METR-LA Traffic Dataset</li>
              <li>时间连续，无断点；207个传感器完整</li>
              <li>0值按疑似缺测候选处理，并保留敏感性对照</li>
              <li>不融合天气等外部数据，避免口径混乱</li>
            </ul>
          </div>
          <div class="dashboard-card">
            <h3>使用边界</h3>
            <ul>
              <li>系统定位是历史数据回放式分析平台</li>
              <li>模型解释反映历史统计关联，不等于因果诊断</li>
              <li>不判断事故、天气等外部原因</li>
              <li>分析结论用于辅助复盘，不替代交通管理决策</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_analysis() -> None:
    st.title("数据分析")
    show_replay_banner()
    hourly = load_csv("reports/eda/tables/hourly_speed_pattern.csv")
    day_type = load_csv("reports/eda/tables/day_type_hour_speed.csv")
    risk = load_csv("reports/eda/tables/sensor_risk_ranking.csv")
    daily = load_csv("reports/eda/tables/daily_speed_trend.csv")
    summary = load_json("reports/eda/eda_summary.json")

    st.subheader("数据概览")
    cols = st.columns(4)
    cols[0].metric("有效速度观测", f"{summary['valid_observations']:,}")
    cols[1].metric("缺测候选", f"{summary['missing_candidate_observations']:,}")
    cols[2].metric("全网同步零值时点", f"{summary['systemwide_zero_timestamps']:,}")
    cols[3].metric("最低速度小时", f"{summary['lowest_network_median_hour']}:00")

    st.subheader("时间规律")
    selected_day_type = st.selectbox("查看日期类型", ["全部", "工作日", "周末"])
    if selected_day_type == "全部":
        chart_data = hourly.set_index("hour")[["median_speed_valid", "median_speed_keep_zero"]]
        chart_data = chart_data.rename(
            columns={
                "median_speed_valid": "剔除0值后的速度中位数",
                "median_speed_keep_zero": "保留0值的速度中位数",
            }
        )
    else:
        day_type_value = {"工作日": "Weekday", "周末": "Weekend"}[selected_day_type]
        filtered = day_type.loc[day_type["day_type"] == day_type_value]
        chart_data = filtered.set_index("hour")[["median_speed_valid"]]
        chart_data = chart_data.rename(columns={"median_speed_valid": "剔除0值后的速度中位数"})
    st.line_chart(chart_data)
    st.markdown(
        '<div class="explain-box">口径说明：METR-LA中有不少速度为0的记录。'
        "它们不一定代表车辆真的完全停止，也可能是传感器异常或短时缺测。"
        "剔除0值后的线用于观察主要交通规律；保留0值的线用于对照数据质量影响。</div>",
        unsafe_allow_html=True,
    )

    st.subheader("道路风险排行")
    top_n = st.slider("排行数量", min_value=5, max_value=20, value=10)
    display_risk = risk.head(top_n)[
        ["sensor_id", "coverage_rate", "congestion_rate_60", "average_event_duration_minutes_60"]
    ].copy()
    display_risk.columns = ["传感器", "有效覆盖率", "描述性拥堵率", "平均事件持续分钟"]
    st.dataframe(display_risk, use_container_width=True, hide_index=True)

    heatmap = load_congestion_heatmap()
    locations = load_sensor_locations()

    st.subheader("道路走廊拥堵热力图")
    if heatmap.empty or locations.empty:
        st.info("本地测试集特征或传感器经纬度文件不可用，暂时无法生成地图热力图。")
    else:
        map_dates = sorted(heatmap["date"].unique())
        map_date = st.selectbox("选择地图日期", map_dates, index=len(map_dates) - 1, key="map_date")
        hour_labels = [f"{hour:02d}:00" for hour in range(24)]
        map_hour_label = st.selectbox("选择地图小时", hour_labels, index=17, key="map_hour")
        map_hour = int(map_hour_label.split(":")[0])
        map_data = heatmap.loc[
            (heatmap["date"] == map_date) & (heatmap["hour"] == map_hour)
        ].merge(locations, how="inner", on="sensor_id")
        if map_data.empty:
            st.info("当前日期和小时没有可展示的传感器点位。")
        else:
            map_data = map_data.copy()
            map_data["congestion_percent"] = map_data["congestion_rate"] * 100
            edges = load_sensor_edges()
            from_points = map_data.rename(
                columns={
                    "sensor_id": "from_sensor_id",
                    "latitude": "from_latitude",
                    "longitude": "from_longitude",
                    "congestion_percent": "from_congestion_percent",
                }
            )[["from_sensor_id", "from_latitude", "from_longitude", "from_congestion_percent"]]
            to_points = map_data.rename(
                columns={
                    "sensor_id": "to_sensor_id",
                    "latitude": "to_latitude",
                    "longitude": "to_longitude",
                    "congestion_percent": "to_congestion_percent",
                }
            )[["to_sensor_id", "to_latitude", "to_longitude", "to_congestion_percent"]]
            edge_data = edges.merge(from_points, how="inner", on="from_sensor_id").merge(
                to_points, how="inner", on="to_sensor_id"
            )
            edge_data["congestion_percent"] = (
                edge_data["from_congestion_percent"] + edge_data["to_congestion_percent"]
            ) / 2
            risk_bands = [
                (0, 20, "0-20% 低拥堵", "#fff7bc", 4),
                (20, 40, "20-40% 轻度拥堵", "#fec44f", 5),
                (40, 60, "40-60% 中度拥堵", "#fe9929", 6),
                (60, 80, "60-80% 高拥堵", "#f03b20", 7),
                (80, 101, "80%以上 严重拥堵", "#bd0026", 8),
            ]
            map_figure = go.Figure()
            for low, high, label, color, width in risk_bands:
                band = edge_data.loc[
                    (edge_data["congestion_percent"] >= low)
                    & (edge_data["congestion_percent"] < high)
                ]
                if band.empty:
                    continue
                latitudes: list[float | None] = []
                longitudes: list[float | None] = []
                customdata: list[list[object] | list[None]] = []
                for row in band.itertuples(index=False):
                    segment_data = [
                        row.from_sensor_id,
                        row.to_sensor_id,
                        float(row.congestion_percent),
                        float(row.distance),
                    ]
                    latitudes.extend([row.from_latitude, row.to_latitude, None])
                    longitudes.extend([row.from_longitude, row.to_longitude, None])
                    customdata.extend([segment_data, segment_data, [None, None, None, None]])
                map_figure.add_trace(
                    go.Scattermapbox(
                        lat=latitudes,
                        lon=longitudes,
                        mode="lines",
                        line={"color": color, "width": width},
                        name=label,
                        customdata=customdata,
                        hovertemplate=(
                            "传感器走廊 %{customdata[0]} → %{customdata[1]}<br>"
                            "时间 " + map_date + f" {map_hour_label}<br>"
                            "走廊拥堵率 %{customdata[2]:.1f}%<br>"
                            "传感器间距 %{customdata[3]:.0f}<extra></extra>"
                        ),
                    )
                )
            map_figure.update_layout(
                height=620,
                mapbox={
                    "style": "carto-positron",
                    "center": {
                        "lat": float(map_data["latitude"].mean()),
                        "lon": float(map_data["longitude"].mean()),
                    },
                    "zoom": 9.3,
                },
                legend={"title": {"text": "走廊拥堵率"}, "orientation": "h", "y": 0.01},
                margin={"l": 0, "r": 0, "t": 10, "b": 10},
            )
            st.plotly_chart(map_figure, use_container_width=True)
            st.caption(
                "说明：线段连接相邻传感器，颜色表示该传感器走廊在所选日期和小时的未来30分钟拥堵率。"
            )
            st.caption(
                "这是基于 METR-LA 传感器坐标和距离关系构建的近似道路走廊，不是官方道路几何线段。"
            )

    st.subheader("道路—小时拥堵热力图")
    heatmap = load_congestion_heatmap()
    if heatmap.empty:
        st.info("本地模型特征数据不存在，暂时无法生成热力图。")
    else:
        # 只展示整体拥堵率最高的道路，保证页面可读；完整数据仍保留在本地结果中。
        dates = sorted(heatmap["date"].unique())
        selected_date = st.selectbox("选择热力图日期", dates, index=len(dates) - 1)
        daily_heatmap = heatmap.loc[heatmap["date"] == selected_date]
        matrix = daily_heatmap.pivot(
            index="sensor_id", columns="hour", values="congestion_rate"
        ).fillna(0)
        top_sensors = matrix.mean(axis=1).sort_values(ascending=False).head(40).index
        display_heatmap = matrix.loc[top_sensors].reindex(columns=range(24), fill_value=0)
        figure = go.Figure(
            go.Heatmap(
                z=display_heatmap.to_numpy() * 100,
                x=[f"{int(hour):02d}:00" for hour in display_heatmap.columns],
                y=display_heatmap.index.astype(str),
                colorscale="YlOrRd",
                colorbar={"title": "拥堵率（%）"},
                hovertemplate="传感器 %{y}<br>时间 %{x}<br>拥堵率 %{z:.1f}%<extra></extra>",
            )
        )
        figure.update_layout(
            height=720,
            xaxis_title="小时",
            yaxis_title="道路传感器（按整体拥堵率排序）",
            margin={"l": 80, "r": 20, "t": 20, "b": 60},
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(f"当前日期：{selected_date}。颜色越深表示该道路在对应小时未来30分钟持续拥堵率越高。")
        st.caption("口径：测试集历史回放样本中，该道路在对应小时未来30分钟发生持续拥堵的比例；颜色越深表示拥堵率越高。")

    st.subheader("速度趋势")
    trend = daily.set_index("date")[["mean_speed_valid", "mean_speed_keep_zero"]]
    trend = trend.rename(
        columns={
            "mean_speed_valid": "剔除0值后的平均速度",
            "mean_speed_keep_zero": "保留0值的平均速度",
        }
    )
    st.line_chart(trend)
    st.markdown(
        '<div class="explain-box">口径说明：剔除0值后的平均速度更接近正常道路运行状态；'
        "保留0值的平均速度会把疑似缺测/异常0值也算进去，因此可能出现突然下跌。"
        "两条线放在一起，是为了说明0值处理会不会明显影响趋势判断。</div>",
        unsafe_allow_html=True,
    )

    st.subheader("拥堵前兆")
    pre_profile = load_csv("reports/eda/tables/pre_congestion_profile.csv")
    pre_figure = go.Figure()
    pre_figure.add_trace(
        go.Scatter(
            x=pre_profile["relative_minutes"],
            y=pre_profile["event_p75_speed_ratio"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    pre_figure.add_trace(
        go.Scatter(
            x=pre_profile["relative_minutes"],
            y=pre_profile["event_p25_speed_ratio"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.18)",
            line={"width": 0},
            name="拥堵事件波动范围",
            hoverinfo="skip",
        )
    )
    pre_figure.add_trace(
        go.Scatter(
            x=pre_profile["relative_minutes"],
            y=pre_profile["event_median_speed_ratio"],
            mode="lines+markers",
            name="拥堵事件速度比中位数",
            line={"color": "#38bdf8", "width": 3},
            marker={"size": 6},
            hovertemplate="相对拥堵起点 %{x} 分钟<br>速度比中位数 %{y:.3f}<extra></extra>",
        )
    )
    pre_figure.add_trace(
        go.Scatter(
            x=pre_profile["relative_minutes"],
            y=pre_profile["control_median_speed_ratio"],
            mode="lines+markers",
            name="正常对照速度比中位数",
            line={"color": "#f97316", "width": 3},
            marker={"size": 6},
            hovertemplate="相对对照时点 %{x} 分钟<br>速度比中位数 %{y:.3f}<extra></extra>",
        )
    )
    pre_figure.add_vline(
        x=0,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="拥堵开始",
        annotation_position="top",
    )
    pre_figure.add_hline(
        y=0.6,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="低速阈值",
        annotation_position="bottom right",
    )
    pre_figure.update_layout(
        title="持续拥堵发生前后的速度变化",
        height=520,
        xaxis_title="距离拥堵开始的时间（分钟）",
        yaxis_title="速度比（当前速度 / 自由流速度）",
        font={"family": "Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif"},
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 20, "r": 20, "t": 70, "b": 95},
    )
    pre_figure.update_yaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(pre_figure, use_container_width=True)
    st.markdown(
        '<div class="explain-box">图中蓝线表示发生持续拥堵的道路在拥堵前后的速度变化；'
        "橙线表示正常对照时段。蓝线在0分钟前持续下降，说明拥堵发生前已经出现明显降速信号。</div>",
        unsafe_allow_html=True,
    )
    st.success(
        f"业务发现：工作日7—9时比周末同期低{abs(summary['weekday_vs_weekend_7_9_speed_difference']):.2f} mph；"
        f"事件前15分钟速度比中位数为{summary['event_profile']['event_ratio_minus_15']:.3f}，"
        f"事件起点降至{summary['event_profile']['event_ratio_at_start']:.3f}。"
    )


def page_prediction() -> None:
    st.title("预测展示")
    show_replay_banner()
    model_path = PROJECT_ROOT / "artifacts/models/xgboost.joblib"
    if not model_path.exists():
        st.warning("最终模型文件不存在。请先完成Track A模型产物准备。")
        return
    model = load_model("artifacts/models/xgboost.joblib")
    summary = load_json("reports/modeling/m3_summary.json")
    selected_result = next(
        item for item in summary["model_results"] if item["model"] == summary["selected_model"]
    )
    threshold = float(selected_result["validation_threshold"]["threshold"])
    risk_table = load_csv("reports/eda/tables/sensor_risk_ranking.csv")
    sensors = risk_table["sensor_id"].astype(str).tolist()
    sensor_id = st.selectbox("道路传感器编号", sensors)
    history = load_sensor_history(sensor_id)
    if history.empty:
        st.warning("该传感器没有可用的历史回放样本。")
        return
    timestamps = history["timestamp"].astype(str).tolist()
    timestamp = st.selectbox("历史回放时点", timestamps, index=len(timestamps) - 1)
    row = history.loc[history["timestamp"].astype(str) == timestamp].iloc[[0]].copy()
    st.caption(f"回放时间：{format_replay_timestamp(timestamp)}")
    probability = predict_history_row(model, row)
    level = risk_level(probability)
    risk_badge(level)
    st.info(f"预测摘要：未来30分钟拥堵概率为 {probability:.1%}，当前风险等级为{level}。")
    ratio = float(row.iloc[0]["current_speed_ratio"])
    state = "当前低速状态" if ratio < 0.60 else "当前非低速状态"

    cols = st.columns(4)
    cols[0].metric("拥堵概率", f"{probability:.1%}")
    cols[1].metric("风险等级", level)
    cols[2].metric("当前速度", f"{float(row.iloc[0]['current_speed_mph']):.2f} mph")
    cols[3].metric("当前速度比", f"{ratio:.3f}")
    st.caption(f"模型阈值：{threshold:.2f}；{state}。模型为历史回放，不代表实时状态。")

    st.subheader("最近60分钟速度比")
    current_index = history.index[history["timestamp"].astype(str) == timestamp][0]
    recent = history.loc[max(0, current_index - 12) : current_index].set_index("timestamp")
    chart = recent[["current_speed_ratio", "lag_speed_ratio_5m"]].rename(
        columns={
            "current_speed_ratio": "当前速度/自由流速度",
            "lag_speed_ratio_5m": "5分钟前速度比",
        }
    )
    st.line_chart(chart)

    st.subheader("主要影响因素")
    explanation = explain_history_row(model, row)
    factors = pd.DataFrame(explanation.get("top_factors", []))
    if factors.empty:
        st.info("当前样本的解释产物不可用。")
    else:
        factors = factors[["feature", "observed_value", "reference_value", "risk_support_delta"]].copy()
        factors = localize_feature_column(factors)
        factors.columns = ["因素", "当前值", "训练参考值", "风险支持变化"]
        factors = format_factor_table(factors.head(5))
        st.dataframe(factors, use_container_width=True, hide_index=True)
        top_factor = factors.iloc[0]
        st.info(
            f"当前最主要信号是“{top_factor['因素']}”，当前值为 "
            f"{top_factor['当前值']}，相对训练参考值 {top_factor['训练参考值']}，"
            f"单因素风险支持变化为 {top_factor['风险支持变化']}。"
        )
        st.caption("风险支持变化为单因素训练参考反事实对照，不是因果效应。")

    st.subheader("历史评估标签")
    label = int(row.iloc[0]["target_congestion_30m"])
    st.info(f"该历史时点的事后评估标签为：{'未来30分钟发生持续拥堵' if label else '未来30分钟未达到持续拥堵定义'}。")


def page_report() -> None:
    st.title("分析报告")
    show_replay_banner()
    summary = load_json("reports/modeling/m3_summary.json")
    metrics = load_csv("reports/modeling/model_comparison.csv")
    importance = load_csv("reports/modeling/feature_importance_xgboost.csv")
    selected = summary["selected_model"]
    selected_result = next(item for item in summary["model_results"] if item["model"] == selected)

    st.subheader("模型评价")
    selected_display = {
        "Logistic Regression": "逻辑回归",
        "Random Forest": "随机森林",
        "XGBoost": "XGBoost",
    }.get(selected, selected)
    st.success(
        f"最终模型：{selected_display}；验证集F1={selected_result['validation']['f1']:.4f}，"
        f"测试集F1={selected_result['test']['f1']:.4f}。"
    )
    st.dataframe(format_model_metrics(metrics, selected), use_container_width=True, hide_index=True)

    model_names = {
        "Logistic Regression": "逻辑回归",
        "Random Forest": "随机森林",
        "XGBoost": "XGBoost",
    }
    metric_labels = {
        "accuracy": "准确率",
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1分数",
        "roc_auc": "ROC-AUC",
    }
    metrics_plot = metrics.copy()
    metrics_plot["model_display"] = metrics_plot["model"].map(model_names).fillna(metrics_plot["model"])
    metric_long = metrics_plot.melt(
        id_vars=["model_display"],
        value_vars=list(metric_labels),
        var_name="metric",
        value_name="score",
    )
    metric_long["metric_display"] = metric_long["metric"].map(metric_labels)
    comparison_figure = go.Figure()
    for metric_key, metric_label in metric_labels.items():
        part = metric_long.loc[metric_long["metric"] == metric_key]
        comparison_figure.add_trace(
            go.Bar(
                x=part["model_display"],
                y=part["score"],
                name=metric_label,
                hovertemplate="模型 %{x}<br>" + metric_label + " %{y:.4f}<extra></extra>",
            )
        )
    comparison_figure.update_layout(
        title="不同模型测试集指标对比",
        height=520,
        barmode="group",
        xaxis_title="模型",
        yaxis_title="指标得分",
        yaxis={"range": [0, 1]},
        font={"family": "Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif"},
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 20, "r": 20, "t": 70, "b": 95},
    )
    st.plotly_chart(comparison_figure, use_container_width=True)

    roc_figure = go.Figure(
        go.Bar(
            x=metrics_plot["model_display"],
            y=metrics_plot["roc_auc"],
            marker_color=["#60a5fa", "#34d399", "#f97316"],
            text=metrics_plot["roc_auc"].map(lambda value: f"{value:.4f}"),
            textposition="outside",
            hovertemplate="模型 %{x}<br>测试集ROC-AUC %{y:.4f}<extra></extra>",
        )
    )
    roc_figure.update_layout(
        title="模型区分能力对比（ROC-AUC）",
        height=420,
        xaxis_title="模型",
        yaxis_title="ROC-AUC 得分",
        yaxis={"range": [0.95, 1.0]},
        font={"family": "Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif"},
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 70, "b": 70},
    )
    st.plotly_chart(roc_figure, use_container_width=True)

    st.subheader("影响因素排名")
    importance = localize_feature_column(importance)
    importance_display = importance.rename(columns={"feature": "影响因素", "importance": "重要性得分"})
    st.dataframe(importance_display.head(15), use_container_width=True, hide_index=True)
    importance_plot = importance_display.head(12).sort_values("重要性得分", ascending=True)
    importance_figure = go.Figure(
        go.Bar(
            x=importance_plot["重要性得分"],
            y=importance_plot["影响因素"],
            orientation="h",
            marker_color="#38bdf8",
            text=importance_plot["重要性得分"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            hovertemplate="影响因素 %{y}<br>重要性得分 %{x:.4f}<extra></extra>",
        )
    )
    importance_figure.update_layout(
        title="XGBoost模型影响因素重要性排名",
        height=560,
        xaxis_title="重要性得分",
        yaxis_title="影响因素",
        font={"family": "Microsoft YaHei, SimHei, Arial Unicode MS, sans-serif"},
        margin={"l": 20, "r": 80, "t": 70, "b": 70},
    )
    st.plotly_chart(importance_figure, use_container_width=True)

    st.subheader("业务建议")
    st.markdown(
        "1. 对历史高风险传感器和工作日高峰时段设置重点复盘与监测清单。\n"
        "2. 当当前速度比和15分钟滚动低点同时恶化时，建议提前关注未来30分钟风险。\n"
        "3. 将历史同期拥堵率作为点位先验，结合当前速度判断，不单独作为干预依据。\n"
        "4. 所有结果需由交通管理人员复核，不把模型解释当作事故或天气因果诊断。"
    )
    st.caption("模型使用固定历史数据、固定阈值和固定版本产物；本页面不重新训练模型。")


def main() -> None:
    st.sidebar.title("交通分析平台")
    st.sidebar.caption("METR-LA 历史回放")
    page = st.sidebar.radio(
        "选择页面",
        ["项目介绍", "数据分析", "预测展示", "分析报告"],
    )
    if page == "项目介绍":
        page_intro()
    elif page == "数据分析":
        page_analysis()
    elif page == "预测展示":
        page_prediction()
    else:
        page_report()


if __name__ == "__main__":
    main()
