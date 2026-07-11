CSS = """
<style>
:root {
  --ink: #16324a;
  --muted: #60798e;
  --line: #dbe8f2;
  --blue-50: #f5f9fd;
  --blue-100: #eaf4fb;
  --blue-200: #d8ebf8;
  --blue-500: #4a91c3;
  --blue-700: #245f88;
}

.stApp { background: #ffffff; color: var(--ink); }
.block-container { max-width: 1180px; padding-top: 3.4rem; padding-bottom: 3rem; }

[data-testid="stHeader"] { background: rgba(255,255,255,.78); backdrop-filter: blur(16px); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0f7fc 0%, #e7f2fa 100%);
  border-right: 1px solid #d4e5f1;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1rem; }
[data-testid="stSidebar"] * { color: #1d4059 !important; }
[data-testid="stSidebar"] h2 { font-size: 1.28rem; letter-spacing: -.02em; margin-bottom: 1.2rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border-radius: 11px; padding: .44rem .58rem; margin: .08rem 0;
  transition: background .18s ease, transform .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.72); transform: translateX(2px); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: #ffffff; box-shadow: 0 6px 18px rgba(49, 100, 138, .10);
}
[data-testid="stSidebar"] hr { border-color: #cbdfea; margin: 1.35rem 0; }

.hero {
  position: relative; overflow: hidden; padding: 2rem 2.15rem; border-radius: 24px;
  background:
    radial-gradient(circle at 88% 18%, rgba(255,255,255,.86) 0, rgba(255,255,255,0) 27%),
    linear-gradient(125deg, #e8f5fc 0%, #c9e5f6 100%);
  border: 1px solid #c9e2f1;
  box-shadow: 0 16px 42px rgba(50, 104, 143, .12);
  color: var(--ink); margin-bottom: 1rem;
}
.hero::after {
  content: ""; position: absolute; width: 190px; height: 190px; right: -58px; bottom: -108px;
  border: 26px solid rgba(67, 139, 187, .09); border-radius: 50%;
}
.hero h1 { position: relative; margin: .55rem 0 .45rem; color: #153c5b; font-size: clamp(2rem, 4vw, 3rem); letter-spacing: -.045em; line-height: 1.13; }
.hero p { position: relative; margin: 0; color: #466a84; font-size: 1.02rem; }
.traffic-motif {
  position: absolute; width: 250px; right: 1.45rem; bottom: .9rem;
  color: #4e91bc; opacity: .62; pointer-events: none;
}
.traffic-motif svg { display: block; width: 100%; height: auto; overflow: visible; }
.traffic-motif .route-shadow { fill: none; stroke: rgba(255,255,255,.72); stroke-width: 14; stroke-linecap: round; }
.traffic-motif .route { fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-dasharray: 5 8; }
.traffic-motif circle { fill: #f8fcff; stroke: currentColor; stroke-width: 2; }
.eyebrow {
  display: inline-flex; align-items: center; padding: .28rem .62rem; border-radius: 999px;
  background: rgba(255,255,255,.68); border: 1px solid rgba(93, 150, 188, .28);
  color: #3f7397; font-weight: 700; letter-spacing: .1em; font-size: .72rem;
}

.notice {
  border: 1px solid #d5e7f3; border-left: 4px solid #4a91c3;
  background: #f7fbfe; padding: .84rem 1rem; border-radius: 12px;
  color: #34566e; margin: .75rem 0 1.15rem;
}

.card {
  height: 100%; min-height: 138px; background: rgba(255,255,255,.96);
  border: 1px solid var(--line); border-radius: 17px; padding: 1.2rem 1.25rem;
  box-shadow: 0 8px 24px rgba(52, 99, 133, .07);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.card:hover { transform: translateY(-2px); border-color: #bdd9ea; box-shadow: 0 14px 30px rgba(52, 99, 133, .11); }
.card-icon {
  position: relative; display: inline-grid; place-items: center; width: 44px; height: 44px; margin-bottom: .85rem;
  border-radius: 13px; color: var(--blue-700);
  background: linear-gradient(145deg, #f8fcff 0%, #dceefa 100%);
  border: 1px solid #d3e7f4;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.95), 0 7px 16px rgba(61, 119, 157, .12);
  transition: transform .2s ease, box-shadow .2s ease;
}
.card-icon::after {
  content: ""; position: absolute; inset: -5px; z-index: -1; border-radius: 16px;
  background: rgba(206, 232, 248, .34); opacity: 0; transform: scale(.86); transition: .2s ease;
}
.card-icon svg { width: 23px; height: 23px; fill: none; stroke: currentColor; stroke-width: 1.75; stroke-linecap: round; stroke-linejoin: round; }
.card:hover .card-icon { transform: translateY(-2px) rotate(-2deg); box-shadow: inset 0 1px 0 #fff, 0 10px 20px rgba(61, 119, 157, .18); }
.card:hover .card-icon::after { opacity: 1; transform: scale(1); }
.card h3 { margin: 0 0 .42rem; color: #235b7a; font-size: 1.08rem; letter-spacing: -.015em; }
.card p { margin: 0; color: #587185; line-height: 1.65; font-size: .92rem; }

.process {
  display: flex; align-items: center; flex-wrap: wrap; gap: .5rem; margin: .3rem 0 1.3rem;
}
.process span { padding: .46rem .74rem; background: var(--blue-50); border: 1px solid var(--line); border-radius: 999px; color: #315e7d; font-size: .9rem; font-weight: 600; }
.process b { color: #8eb6cf; font-weight: 500; }

.empty {
  text-align: center; padding: 2.35rem 1.4rem; border: 1px dashed #9fc2d9;
  border-radius: 18px; background: linear-gradient(180deg, #fbfdff, #f5fafe); color: #486b82;
}
.empty strong { display: inline-block; color: #245f88; font-size: 1.05rem; }
.footer { color: #708696; font-size: .8rem; margin-top: 2.3rem; padding-top: 1rem; border-top: 1px solid #edf2f5; }

[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stSelectbox"] > div, [data-testid="stSlider"] { border-radius: 12px; }
button[kind="header"] { border-radius: 10px; }
@media (max-width: 900px) {
  .traffic-motif { width: 190px; right: -.5rem; opacity: .25; }
  .hero h1, .hero p, .eyebrow { z-index: 1; }
}
</style>
"""
