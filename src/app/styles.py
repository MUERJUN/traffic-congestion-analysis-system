CSS = """
<style>
:root {
  --bg: #0b111d;
  --panel: #111827;
  --panel-soft: #162033;
  --panel-line: #243249;
  --text: #f8fafc;
  --muted: #9aa8bb;
  --blue: #38bdf8;
  --blue-soft: rgba(56, 189, 248, .12);
  --green: #22c55e;
  --amber: #f59e0b;
  --red: #ef4444;
}

.stApp {
  background:
    radial-gradient(circle at 86% 6%, rgba(56, 189, 248, .08), transparent 28rem),
    linear-gradient(180deg, #0b111d 0%, #080d17 100%);
  color: var(--text);
}

[data-testid="stHeader"] {
  background: rgba(11, 17, 29, .74);
  backdrop-filter: blur(14px);
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
}

.block-container {
  max-width: 1280px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
  width: 15.5rem !important;
  min-width: 15.5rem !important;
  max-width: 15.5rem !important;
  background: linear-gradient(180deg, #121827 0%, #0f1522 100%);
  border-right: 1px solid rgba(148, 163, 184, .14);
}
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  width: 15.5rem !important;
}
[data-testid="stSidebar"] > div:first-child {
  width: 15.5rem !important;
  padding: 1.35rem .85rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
  color: #eef2ff !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
  font-size: 1.05rem !important;
  line-height: 1.18 !important;
  letter-spacing: -.01em;
  margin: .15rem 0 .25rem !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #7f8da3 !important;
  font-size: .74rem !important;
  margin-bottom: 1.25rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  font-size: .82rem;
  color: #b6c2d4;
  margin-bottom: .35rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  min-height: 2.15rem;
  border-radius: 10px;
  padding: .34rem .58rem;
  margin: .12rem 0;
  background: transparent;
  border: 1px solid transparent;
  transition: background .16s ease, border-color .16s ease, transform .16s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: rgba(148, 163, 184, .08);
  border-color: rgba(148, 163, 184, .10);
  transform: translateX(1px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(56, 189, 248, .18), rgba(56, 189, 248, .05));
  border-color: rgba(56, 189, 248, .28);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span {
  color: #e0f2fe !important;
  font-weight: 700;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
  width: 0 !important;
  min-width: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
  font-size: .93rem !important;
  line-height: 1.2 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
  display: none !important;
}

h1, [data-testid="stMarkdownContainer"] h1 {
  font-size: clamp(2rem, 3vw, 3.25rem) !important;
  line-height: 1.12 !important;
  letter-spacing: -.045em !important;
  margin-bottom: 1rem !important;
}
h2, [data-testid="stMarkdownContainer"] h2 {
  font-size: 1.45rem !important;
  line-height: 1.22 !important;
  letter-spacing: -.025em !important;
  margin-top: 1.4rem !important;
}
h3, [data-testid="stMarkdownContainer"] h3 {
  font-size: 1.08rem !important;
  line-height: 1.25 !important;
}
p, li, [data-testid="stMarkdownContainer"] {
  font-size: .98rem;
  line-height: 1.7;
}

.replay-banner {
  padding: .65rem .85rem;
  border-radius: 10px;
  background: rgba(245, 158, 11, .10);
  border: 1px solid rgba(245, 158, 11, .28);
  color: #fde68a;
  margin: .25rem 0 1.1rem;
}

.hero-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--panel-line);
  border-radius: 18px;
  padding: 1.4rem 1.55rem;
  background:
    linear-gradient(135deg, rgba(56, 189, 248, .16), rgba(34, 197, 94, .06)),
    linear-gradient(180deg, rgba(17, 24, 39, .96), rgba(15, 23, 42, .96));
  box-shadow: 0 16px 42px rgba(0, 0, 0, .22);
  margin-bottom: 1rem;
}
.hero-panel::after {
  content: "";
  position: absolute;
  inset: auto -7rem -8rem auto;
  width: 18rem;
  height: 18rem;
  border-radius: 999px;
  border: 2.2rem solid rgba(56, 189, 248, .07);
}
.hero-panel .eyebrow {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  border-radius: 999px;
  padding: .28rem .65rem;
  color: #bae6fd;
  background: rgba(56, 189, 248, .10);
  border: 1px solid rgba(56, 189, 248, .22);
  font-size: .78rem;
  font-weight: 700;
}
.hero-panel h1 {
  max-width: 950px;
  margin: .9rem 0 .65rem !important;
}
.hero-panel p {
  max-width: 980px;
  color: #cbd5e1;
  margin: 0;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
  margin: 1rem 0 1.1rem;
}
.kpi-card {
  border: 1px solid var(--panel-line);
  border-radius: 14px;
  padding: .9rem 1rem;
  background: rgba(17, 24, 39, .88);
}
.kpi-card span {
  display: block;
  color: var(--muted);
  font-size: .78rem;
}
.kpi-card strong {
  display: block;
  margin-top: .25rem;
  color: var(--text);
  font-size: 1.35rem;
  line-height: 1.1;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .9rem;
  margin: 1rem 0;
}
.dashboard-card {
  border: 1px solid var(--panel-line);
  border-radius: 16px;
  padding: 1rem 1.05rem;
  background: rgba(17, 24, 39, .82);
  min-height: 160px;
}
.dashboard-card h3 {
  margin: 0 0 .6rem;
  color: #e0f2fe;
}
.dashboard-card ul {
  margin: .3rem 0 0 1.05rem;
  padding: 0;
  color: #d6deea;
}
.dashboard-card li {
  margin: .25rem 0;
}

.process-bar {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
  margin-top: .6rem;
}
.process-bar span {
  border: 1px solid rgba(56, 189, 248, .22);
  border-radius: 999px;
  padding: .35rem .6rem;
  color: #c7e9ff;
  background: rgba(56, 189, 248, .08);
  font-size: .86rem;
}

.explain-box {
  padding: .72rem .85rem;
  border-left: 4px solid var(--blue);
  border-radius: 10px;
  background: var(--blue-soft);
  color: #d8ecff;
  margin: .55rem 0 1rem;
}

[data-testid="stMetric"] {
  border: 1px solid var(--panel-line);
  border-radius: 14px;
  padding: .85rem .95rem;
  background: rgba(17, 24, 39, .78);
}
[data-testid="stMetricLabel"] p {
  color: var(--muted) !important;
  font-size: .82rem !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.35rem !important;
}

[data-testid="stAlert"] {
  border-radius: 12px;
}
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
[data-testid="stImage"] {
  border-radius: 14px;
}

@media (max-width: 900px) {
  .block-container {
    padding-top: 1.4rem;
  }
  .kpi-grid,
  .dashboard-grid {
    grid-template-columns: 1fr;
  }
}
</style>
"""
