from pathlib import Path
import streamlit as st
from .artifacts import load_json_artifact
from .styles import CSS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
PAGES = ("项目介绍", "数据分析", "预测展示", "分析报告")
DATA_RANGE = "2012-03-01 00:00 — 2012-06-27 23:55"

def hero(title: str, subtitle: str, traffic_motif: bool = False) -> None:
    motif = ""
    if traffic_motif:
        motif = '''<div class="traffic-motif" aria-hidden="true"><svg viewBox="0 0 260 150"><path class="route-shadow" d="M18 116 C64 50 100 132 145 65 S222 28 246 45"/><path class="route" d="M18 116 C64 50 100 132 145 65 S222 28 246 45"/><circle cx="18" cy="116" r="5"/><circle cx="145" cy="65" r="5"/><circle cx="246" cy="45" r="5"/></svg></div>'''
    st.markdown(f'<div class="hero"><div class="eyebrow">METR-LA · HISTORICAL REPLAY</div><h1>{title}</h1><p>{subtitle}</p>{motif}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="notice"><strong>历史数据回放，非实时系统</strong>　数据范围：{DATA_RANGE}。页面结果仅用于课程研究与辅助分析。</div>', unsafe_allow_html=True)

def empty(title: str, detail: str) -> None:
    st.markdown(f'<div class="empty"><strong>{title}</strong><br><br>{detail}</div>', unsafe_allow_html=True)

def introduction() -> None:
    hero("城市道路拥堵风险分析", "从数据质量、规律分析到未来 30 分钟风险解释的完整证据链", traffic_motif=True)
    sensor_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="2.5"/><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.9 4.9a10 10 0 0 0 0 14.2M19.1 4.9a10 10 0 0 1 0 14.2"/></svg>'
    clock_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2M7.5 3.8 5.6 5.5M16.5 3.8l1.9 1.7"/></svg>'
    trend_icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 17.5 9 12l3.5 3 7-8"/><path d="M14.5 7H19v4.5"/><path d="M4 20h16"/></svg>'
    cards = ((sensor_icon, "207 个观测点", "METR-LA 高速公路传感器，不将传感器直接等同于完整城市道路。"), (clock_icon, "5 分钟粒度", "连续历史速度观测，用于分析周期规律、趋势与历史回放案例。"), (trend_icon, "未来 30 分钟", "正式模型结果只读取 Track A 固化的版本化产物。"))
    for col, (icon, title, body) in zip(st.columns(3), cards):
        col.markdown(f'<div class="card"><div class="card-icon">{icon}</div><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.subheader("分析闭环")
    st.markdown('<div class="process"><span>数据审计</span><b>→</b><span>探索分析</span><b>→</b><span>风险预测</span><b>→</b><span>模型解释</span><b>→</b><span>业务建议</span></div>', unsafe_allow_html=True)
    st.subheader("使用指引")
    st.markdown("1. 在“数据分析”查看时段规律、观测点排行和数据质量。\n2. 在“预测展示”选择合法传感器与可用历史时点。\n3. 在“分析报告”核对模型指标、解释依据与限制。")
    st.info("能力边界：单一历史数据源；不接入实时交通；不诊断事故、天气或施工等因果事件。")

def analysis() -> None:
    hero("数据分析", "识别历史速度规律、重点观测点与拥堵前兆")
    result = load_json_artifact(ARTIFACTS / "eda" / "dashboard_summary.json", {"schema_version", "generated_at", "data_version", "overview", "time_patterns", "sensor_rankings"})
    with st.sidebar:
        st.markdown("### 分析筛选")
        st.selectbox("日期类型", ("全部", "工作日", "周末"), disabled=not result.available)
        st.slider("排行数量", 5, 30, 10, disabled=not result.available)
    if not result.available:
        empty("正式 EDA 汇总尚未接入", result.error or "无法加载分析产物。")
        st.caption("计划展示：数据概览、小时曲线、星期×小时热力图、道路排行、速度趋势和拥堵前兆。")
        return
    st.success("EDA 产物已通过模式与字段校验。")
    st.json(result.data, expanded=False)

def prediction() -> None:
    hero("预测展示", "选择历史时点，查看未来 30 分钟拥堵风险及模型依据")
    result = load_json_artifact(ARTIFACTS / "predictions" / "replay_cases.json", {"schema_version", "generated_at", "data_version", "model_version", "cases"})
    st.selectbox("道路传感器编号", ("等待可用传感器列表",), disabled=True)
    st.selectbox("历史回放时点", ("等待有效历史时点",), disabled=True)
    if not result.available:
        empty("当前无法生成预测", result.error or "预测产物不可用。")
        st.warning("页面不会用默认概率、随机数字或静态演示指标代替正式模型结果。")
        return
    st.success("预测产物已通过模式与字段校验。")
    st.json(result.data, expanded=False)

def report() -> None:
    hero("分析报告", "核对模型比较、解释证据、误差表现与适用限制")
    result = load_json_artifact(ARTIFACTS / "metrics" / "model_comparison.json", {"schema_version", "generated_at", "data_version", "models", "selected_model", "label_definition"})
    if not result.available:
        empty("正式模型报告尚未生成", result.error or "模型比较产物不可用。")
        st.caption("待展示：三模型五项指标、混淆矩阵、ROC 曲线、全局重要性和分组误差分析。")
        return
    st.success("模型比较产物已通过模式与字段校验。")
    st.json(result.data, expanded=False)

def run() -> None:
    st.set_page_config(page_title="交通拥堵历史回放", page_icon="🚦", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 交通风险分析")
        page = st.radio("页面导航", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("数据：METR-LA · 2012")
        st.caption("历史回放 · 非实时系统")
    {"项目介绍": introduction, "数据分析": analysis, "预测展示": prediction, "分析报告": report}[page]()
    st.markdown('<div class="footer">所有模型解释均表示统计关联或模型判断依据，不构成交通事件因果诊断。</div>', unsafe_allow_html=True)
