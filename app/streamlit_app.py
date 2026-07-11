"""Historical METR-LA congestion analysis and risk replay dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.dashboard_utils import (  # noqa: E402
    ROOT as PROJECT_ROOT,
    explain_history_row,
    load_csv,
    load_image_bytes,
    load_json,
    load_model,
    load_sensor_history,
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
    .replay-banner { padding: 0.65rem 1rem; border-radius: 0.5rem; background: #fff4ce; color: #6b4e00; margin-bottom: 1rem; }
    .metric-note { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_replay_banner() -> None:
    st.markdown(
        '<div class="replay-banner">历史交通数据回放平台：页面结果来自METR-LA历史样本，不是实时交通系统。</div>',
        unsafe_allow_html=True,
    )


def page_intro() -> None:
    st.title("基于机器学习的城市道路拥堵风险预测与分析辅助系统")
    show_replay_banner()
    st.write(
        "本系统将METR-LA道路传感器速度数据转化为时间规律、风险预测、模型解释和交通建议，"
        "用于课程项目分析与历史案例复盘。"
    )
    left, right = st.columns(2)
    with left:
        st.subheader("项目闭环")
        st.markdown("问题提出 → 数据审计 → EDA → 历史特征 → 模型比较 → 解释 → 建议")
        st.subheader("数据基线")
        st.markdown(
            "- 207个道路传感器\n"
            "- 五分钟采样\n"
            "- 2012-03-01至2012-06-27\n"
            "- 0值按缺测候选处理"
        )
    with right:
        st.subheader("系统能力")
        st.markdown(
            "- 查看时间规律和道路风险排行\n"
            "- 选择历史传感器时点回放风险预测\n"
            "- 查看拥堵概率、风险等级和主要因素\n"
            "- 查看三模型评价与业务建议"
        )
        st.subheader("使用边界")
        st.info("模型反映历史统计关联，不诊断事故或天气原因，也不替代交通管理人员决策。")


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
    selected_day_type = st.selectbox("查看日期类型", ["全部", "Weekday", "Weekend"])
    if selected_day_type == "全部":
        chart_data = hourly.set_index("hour")[["median_speed_valid", "median_speed_keep_zero"]]
    else:
        filtered = day_type.loc[day_type["day_type"] == selected_day_type]
        chart_data = filtered.set_index("hour")[["median_speed_valid"]]
    st.line_chart(chart_data)
    st.caption("主线把0值作为缺测候选；橙色/对照口径保留0值，仅用于观察数据质量影响。")

    st.subheader("道路风险排行")
    top_n = st.slider("排行数量", min_value=5, max_value=20, value=10)
    display_risk = risk.head(top_n)[
        ["sensor_id", "coverage_rate", "congestion_rate_60", "average_event_duration_minutes_60"]
    ].copy()
    display_risk.columns = ["传感器", "有效覆盖率", "描述性拥堵率", "平均事件持续分钟"]
    st.dataframe(display_risk, use_container_width=True, hide_index=True)

    st.subheader("速度趋势")
    trend = daily.set_index("date")[["mean_speed_valid", "mean_speed_keep_zero"]]
    st.line_chart(trend)
    st.caption("有效速度均值与保留0均值使用相同统计量进行敏感性对照。")

    st.subheader("拥堵前兆")
    st.image(
        load_image_bytes("reports/eda/figures/pre_congestion_profile.png"),
        caption="持续拥堵事件起点前后的速度比中位数",
        use_container_width=True,
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
    probability = predict_history_row(model, row)
    level = risk_level(probability)
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
    st.line_chart(recent[["current_speed_ratio", "lag_speed_ratio_5m"]])

    st.subheader("主要影响因素")
    explanation = explain_history_row(model, row)
    factors = pd.DataFrame(explanation.get("top_factors", []))
    if factors.empty:
        st.info("当前样本的解释产物不可用。")
    else:
        factors = factors[["feature", "observed_value", "reference_value", "risk_support_delta"]].copy()
        factors.columns = ["因素", "当前值", "训练参考值", "风险支持变化"]
        st.dataframe(factors.head(5), use_container_width=True, hide_index=True)
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
    st.success(
        f"最终模型：{selected}；Validation F1={selected_result['validation']['f1']:.4f}，"
        f"Test F1={selected_result['test']['f1']:.4f}。"
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.image(load_image_bytes("reports/modeling/figures/model_comparison.png"), use_container_width=True)
    st.image(load_image_bytes("reports/modeling/figures/test_roc_curves.png"), use_container_width=True)

    st.subheader("影响因素排名")
    st.dataframe(importance.head(15), use_container_width=True, hide_index=True)
    st.image(
        load_image_bytes("reports/modeling/figures/feature_importance_xgboost.png"),
        use_container_width=True,
    )

    st.subheader("业务建议")
    st.markdown(
        "1. 对历史高风险传感器和工作日高峰时段设置重点复盘与监测清单。\n"
        "2. 当当前速度比和15分钟滚动低点同时恶化时，建议提前关注未来30分钟风险。\n"
        "3. 将历史同期拥堵率作为点位先验，结合当前速度判断，不单独作为干预依据。\n"
        "4. 所有结果需由交通管理人员复核，不把模型解释当作事故或天气因果诊断。"
    )
    st.caption("模型使用固定历史数据、固定阈值和固定版本产物；本页面不重新训练模型。")


def main() -> None:
    st.sidebar.title("系统导航")
    st.sidebar.caption("METR-LA历史交通数据回放")
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
