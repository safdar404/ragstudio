"""Vibrant rainbow AI theme — one stylesheet for the whole app (presentation only)."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO_PATH = ASSETS / "agent_logo.svg"

RAINBOW = ["#ff4d6d", "#ff9f1c", "#ffd60a", "#2ee59d", "#22d3ee", "#6366f1", "#c084fc"]


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    try:
        return "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:  # asset missing: fall back to an emoji-only brand
        return ""


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  color-scheme:dark;
  --bg:#0a0a1c; --bg2:#100e2a; --surface:#16133a; --surface2:#1d1946; --line:#2f2a66; --line2:#433c8a;
  --ink:#f4f2ff; --soft:#d9d4ff; --muted:#a9a3d9;
  --red:#ff4d6d; --orange:#ff9f1c; --yellow:#ffd60a; --green:#2ee59d; --cyan:#22d3ee; --indigo:#6366f1; --violet:#c084fc; --pink:#ff4d8d;
  --rainbow:linear-gradient(90deg,#ff4d6d,#ff9f1c,#ffd60a,#2ee59d,#22d3ee,#6366f1,#c084fc);
  --rainbow-d:linear-gradient(120deg,#ff4d6d,#ff9f1c 18%,#ffd60a 34%,#2ee59d 50%,#22d3ee 66%,#6366f1 82%,#c084fc);
  --aurora:linear-gradient(110deg,#6366f1,#c084fc 45%,#ff4d8d);
  --glow:0 0 0 1px #ffffff10,0 10px 30px -12px #c084fc99;
  --shadow:0 1px 0 #ffffff0d inset,0 14px 30px -20px #000;
  --r:14px;
}
html,body,.stApp,[class*="css"],button,input,textarea,select{font-family:"Plus Jakarta Sans",Inter,ui-sans-serif,system-ui,"Segoe UI",sans-serif}
.stApp{color:var(--ink);font-size:14.5px;
  background:
    radial-gradient(700px 380px at 0% 0%,#ff4d6d22,transparent 60%),
    radial-gradient(700px 380px at 100% 0%,#22d3ee22,transparent 60%),
    radial-gradient(800px 420px at 50% 100%,#c084fc26,transparent 60%),
    radial-gradient(500px 300px at 85% 60%,#2ee59d14,transparent 60%),
    var(--bg);background-attachment:fixed}
[data-testid="stHeader"]{background:#0a0a1ccc;backdrop-filter:blur(12px)}
[data-testid="stHeader"]:after{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:var(--rainbow);opacity:.85}
.block-container{max-width:1320px;padding:3.6rem clamp(14px,2.4vw,36px) 3rem!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{color:#fff;letter-spacing:-.02em}
.stApp p,.stApp li{line-height:1.6}
.stApp label,.stApp summary,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted)!important}
[data-testid="stVerticalBlock"]{gap:.7rem}
::selection{background:#c084fc66}
:focus-visible{outline:2px solid var(--cyan)!important;outline-offset:2px!important}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#121034,#0c0a24)!important;border-right:1px solid var(--line)}
[data-testid="stSidebar"]:before{content:"";position:absolute;top:0;bottom:0;right:0;width:2px;background:linear-gradient(180deg,#ff4d6d,#ffd60a,#2ee59d,#22d3ee,#c084fc);opacity:.6}
[data-testid="stSidebarUserContent"]{padding-top:.4rem}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important;box-shadow:none!important}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{font-weight:700;font-size:13.5px;padding:.45rem .8rem}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.55rem}
.side-brand{display:flex;align-items:center;gap:11px;margin:0 0 .9rem;padding:10px 12px;border-radius:16px;
  background:linear-gradient(var(--surface),var(--surface)) padding-box,var(--rainbow-d) border-box;border:1.5px solid transparent}
.side-brand img{width:46px;height:46px;flex:none;filter:drop-shadow(0 6px 14px #c084fc66)}
.side-brand b{display:block;font-size:16px;font-weight:800;line-height:1.1;background:var(--rainbow);-webkit-background-clip:text;background-clip:text;color:transparent}
.side-brand span{display:block;font-size:10.5px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;margin-top:3px}
.usage{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:.2rem 0 .3rem}
.usage div{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:7px 10px;position:relative;overflow:hidden}
.usage div:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px}
.usage div:nth-child(1):before{background:var(--red)}.usage div:nth-child(2):before{background:var(--yellow)}
.usage div:nth-child(3):before{background:var(--green)}.usage div:nth-child(4):before{background:var(--cyan)}
.usage small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.usage b{font-family:"JetBrains Mono",monospace;font-size:13px;color:#fff}

/* ── Hero ── */
.hero{position:relative;display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
  padding:18px 22px;border-radius:20px;margin-bottom:12px;isolation:isolate;overflow:hidden;
  background:linear-gradient(135deg,#1b1747f2,#120f33f2) padding-box,var(--rainbow-d) border-box;border:1.5px solid transparent;
  box-shadow:0 20px 50px -30px #c084fc}
.hero:before{content:"";position:absolute;inset:-60%;z-index:-1;opacity:.45;filter:blur(40px);
  background:conic-gradient(from 0deg,#ff4d6d55,#ffd60a44,#2ee59d44,#22d3ee55,#6366f155,#c084fc55,#ff4d6d55);
  animation:spin 26s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.hero-main{display:flex;align-items:center;gap:16px;min-width:0}
.hero-logo{width:68px;height:68px;flex:none;filter:drop-shadow(0 10px 20px #c084fc80);animation:float 5s ease-in-out infinite}
@keyframes float{50%{transform:translateY(-4px)}}
.eyebrow{display:inline-flex;align-items:center;gap:7px;font-size:10.5px;letter-spacing:.18em;font-weight:800;color:#fff;
  padding:3px 10px;border-radius:999px;background:#ffffff12;border:1px solid #ffffff26}
.eyebrow i{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 2s infinite}
@keyframes pulse{50%{opacity:.3}}
.hero h1{margin:6px 0 2px!important;padding:0!important;font-size:clamp(22px,2.5vw,30px)!important;font-weight:800!important;line-height:1.15!important}
.hero h1 span{background:var(--rainbow);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shift 8s linear infinite}
@keyframes shift{to{background-position:200% 0}}
.hero p{margin:0;color:var(--soft);font-size:13.5px}
.hero.big{padding:28px 30px}
.hero.big .hero-logo{width:96px;height:96px}
.hero.big h1{font-size:clamp(28px,3.4vw,40px)!important}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.chips span{padding:4px 11px;border-radius:999px;font-size:12px;font-weight:600;color:#fff;background:#ffffff0d;border:1px solid}
.chips span:nth-child(7n+1){border-color:#ff4d6d99}.chips span:nth-child(7n+2){border-color:#ff9f1c99}
.chips span:nth-child(7n+3){border-color:#ffd60a99}.chips span:nth-child(7n+4){border-color:#2ee59d99}
.chips span:nth-child(7n+5){border-color:#22d3ee99}.chips span:nth-child(7n+6){border-color:#6366f199}.chips span:nth-child(7n){border-color:#c084fc99}
.stats{display:flex;gap:8px;flex-wrap:wrap}
.stat{min-width:86px;padding:8px 12px;border-radius:14px;background:#ffffff0b;border:1px solid #ffffff1f;text-align:left}
.stat b{display:block;font-size:20px;font-weight:800;line-height:1.1;font-family:"JetBrains Mono",monospace}
.stat:nth-child(1) b{color:var(--yellow)}.stat:nth-child(2) b{color:var(--green)}.stat:nth-child(3) b{color:var(--cyan)}.stat:nth-child(4) b{color:var(--violet)}
.stat small{font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"],[data-baseweb="tab-list"]{gap:4px;padding:5px;border:1px solid var(--line)!important;border-radius:14px;
  background:var(--bg2);width:fit-content;max-width:100%;overflow-x:auto;margin-bottom:.5rem}
[role="tab"],[data-baseweb="tab"]{padding:6px 14px!important;border-radius:10px!important;font-weight:700;color:var(--muted)!important;height:auto!important;min-height:36px;transition:background .2s,color .2s;white-space:nowrap}
[role="tab"] p{font-size:13.5px!important;font-weight:700}
[role="tab"]:hover{color:#fff!important;background:#ffffff0d}
[role="tab"][aria-selected="true"],[data-baseweb="tab"][aria-selected="true"]{background:var(--aurora)!important;color:#fff!important;box-shadow:var(--glow)}
[role="tab"][aria-selected="true"] p{color:#fff!important}
[data-testid="stTab"] > div:not([data-testid]),[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}

/* ── Cards & containers ── */
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:var(--r)!important;box-shadow:var(--shadow)}
[data-testid="stExpander"] summary{font-weight:700}
[data-testid="stExpander"] summary:hover{color:var(--cyan)!important}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--r)}
[data-testid="stForm"]{background:var(--surface);border:1px solid var(--line)!important;border-radius:var(--r);padding:16px 18px!important}
.st-key-connect_card{background:linear-gradient(var(--surface),var(--surface)) padding-box,var(--rainbow-d) border-box!important;border:1.5px solid transparent!important;border-radius:18px!important;padding:18px 20px!important;box-shadow:0 20px 50px -30px #22d3ee}
[class*="st-key-doc_"]{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px!important;position:relative;overflow:hidden}
[class*="st-key-doc_"]:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#ff4d6d,#ffd60a,#2ee59d,#22d3ee,#c084fc)}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px}
[data-testid="stMetricValue"]{font-family:"JetBrains Mono",monospace;font-size:24px!important;color:#fff}
[data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"]{border-top:3px solid var(--red)}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"]{border-top:3px solid var(--yellow)}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"]{border-top:3px solid var(--green)}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"]{border-top:3px solid var(--cyan)}
[data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"]{border-top:3px solid var(--violet)}

/* ── Chat ── */
[data-testid="stChatMessage"]{background:var(--surface)!important;border:1px solid var(--line);border-radius:16px;padding:12px 16px;margin-bottom:4px;position:relative;overflow:hidden}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"] img):before,
[data-testid="stChatMessage"]:has(img[alt="assistant avatar"]):before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#ff4d6d,#ffd60a,#2ee59d,#22d3ee,#c084fc)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]){background:linear-gradient(135deg,#231d55,#1a1645)!important;border-color:#433c8a}
[data-testid="stChatMessageAvatarCustom"] img,[data-testid="stChatMessage"] img{border-radius:10px}
[data-testid="stChatMessageAvatarUser"]{background:var(--aurora)!important;color:#fff}
[data-testid="stChatInput"]{border-radius:16px!important;border:1.5px solid transparent!important;
  background:linear-gradient(var(--bg2),var(--bg2)) padding-box,var(--rainbow) border-box!important;box-shadow:0 12px 30px -18px #22d3ee}
[data-testid="stChatInput"] textarea{color:#fff!important;caret-color:var(--cyan)}
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:#0a0a1cf0!important}

/* ── Inputs ── */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input{background:var(--bg2)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;caret-color:var(--cyan)}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="textarea"]{background:var(--bg2)!important;border-color:var(--line2)!important}
input::placeholder,textarea::placeholder{color:#8f89c4!important;-webkit-text-fill-color:#8f89c4!important}
.stTextInput input:focus,.stTextArea textarea:focus{box-shadow:0 0 0 3px #22d3ee33!important}
[data-baseweb="tag"]{background:#6366f133!important;border:1px solid #6366f188}
[data-baseweb="tag"] span{color:#e0e0ff!important}
[data-testid="stFileUploaderDropzone"]{background:linear-gradient(120deg,#ff4d6d0f,#ffd60a0d,#2ee59d0f,#22d3ee12,#c084fc14)!important;border:1.5px dashed #6d64c9!important;border-radius:14px!important;padding:14px!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cyan)!important}

/* ── Buttons ── */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button,[data-testid="stPopover"] button{border-radius:11px!important;font-weight:650;min-height:36px;padding:4px 12px!important;
  background:var(--surface2)!important;color:#fff!important;border:1px solid var(--line2)!important;transition:transform .15s,border-color .15s,box-shadow .15s}
.stButton>button p,.stDownloadButton>button p{font-size:13.5px}
@media(hover:hover){.stButton>button:not(:disabled):hover,.stDownloadButton>button:not(:disabled):hover,[data-testid="stPopover"] button:hover{
  transform:translateY(-1px);border-color:var(--cyan)!important;box-shadow:0 8px 20px -12px #22d3ee}}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--rainbow-d)!important;background-size:180% 100%!important;border:0!important;color:#160b2e!important;font-weight:800!important;box-shadow:0 10px 26px -12px #ff9f1c}
button[kind="primary"] p,button[kind="primaryFormSubmit"] p{color:#160b2e!important;font-weight:800}
button[kind="primary"]:not(:disabled):hover{background-position:100% 0!important}
button[kind="tertiary"]{background:transparent!important;border:0!important;color:var(--cyan)!important;min-height:30px}
button:disabled{opacity:.45!important;transform:none!important;box-shadow:none!important}
/* Agent quick actions: rainbow-tinted chips */
[class*="st-key-qa_"] button{border-radius:999px!important;min-height:32px;padding:2px 12px!important;background:#ffffff08!important}
[class*="st-key-qa_"] button p{font-size:12.5px!important;white-space:nowrap}
.st-key-qa_0 button{border-color:#ff4d6d88!important}.st-key-qa_1 button{border-color:#ff9f1c88!important}
.st-key-qa_2 button{border-color:#ffd60a88!important}.st-key-qa_3 button{border-color:#2ee59d88!important}
.st-key-qa_4 button{border-color:#22d3ee88!important}.st-key-qa_5 button{border-color:#c084fc88!important}
[class*="st-key-sg_"] button{text-align:left;justify-content:flex-start;background:#ffffff06!important;border-style:dashed!important}
[class*="st-key-sg_"] button p{font-size:12.5px!important}

/* ── Alerts ── */
[data-testid="stAlert"]{border-radius:12px!important}
[data-testid="stAlert"] p{color:#fff!important}

/* ── Pills, empty states, source cards ── */
.pills{display:flex;gap:6px;flex-wrap:wrap;margin:.3rem 0 .1rem}
.pill{font-size:11.5px;padding:2px 9px;border-radius:999px;background:#ffffff0a;border:1px solid var(--line2);color:var(--soft)}
.pill.ok{background:#2ee59d1a;border-color:#2ee59d77;color:#a7f5d3}
.pill.warn{background:#ff9f1c1a;border-color:#ff9f1c77;color:#ffd59a}
.pill.info{background:#22d3ee1a;border-color:#22d3ee77;color:#a5f0fb}
.pill.kind{background:var(--aurora);color:#fff;border:0;font-weight:700}
.empty{text-align:center;padding:30px 18px;border:1.5px dashed var(--line2);border-radius:18px;background:var(--surface)}
.empty .icon{display:inline-grid;place-items:center;width:58px;height:58px;border-radius:18px;background:var(--rainbow-d);font-size:28px;box-shadow:0 10px 30px -12px #c084fc}
.empty h3{margin:.7rem 0 .2rem!important;font-size:18px!important}
.empty p{color:var(--muted);margin:0}
.src{border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:7px 0;background:var(--bg2)}
.src.cited{background:linear-gradient(var(--bg2),var(--bg2)) padding-box,var(--rainbow) border-box;border:1.5px solid transparent}
.src-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.src-id{font-family:"JetBrains Mono",monospace;font-weight:700;font-size:11.5px;padding:1px 7px;border-radius:6px;background:var(--line2);color:#fff}
.src.cited .src-id{background:var(--aurora)}
.src-file{font-weight:700;color:#fff}
.src-loc{color:var(--muted)}
.src-tag{margin-left:auto;font-size:11px;font-weight:700;color:#160b2e;background:var(--green);padding:1px 8px;border-radius:999px}
.src-body{margin-top:7px;font-size:13px;line-height:1.55;color:var(--soft);white-space:pre-wrap;max-height:200px;overflow:auto}
.src-meta{display:flex;gap:12px;align-items:center;margin-top:8px;font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--muted);flex-wrap:wrap}
.bar{flex:1;min-width:80px;max-width:180px;height:5px;background:#ffffff14;border-radius:9px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--rainbow)}
mark{background:linear-gradient(90deg,#ffd60a,#ff9f1c);color:#1a0f00;padding:0 2px;border-radius:3px}
[data-testid="stMarkdownContainer"] code{color:#a5f0fb;background:#22d3ee1a;border-radius:6px;padding:1px 6px}
[data-testid="stCode"],[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
.kw{display:flex;flex-wrap:wrap;gap:5px;margin:.2rem 0 .4rem}
.kw span{font-size:12px;padding:2px 9px;border-radius:999px;color:#fff;font-weight:600}
.kw span:nth-child(7n+1){background:#ff4d6d33}.kw span:nth-child(7n+2){background:#ff9f1c33}.kw span:nth-child(7n+3){background:#ffd60a2e}
.kw span:nth-child(7n+4){background:#2ee59d2e}.kw span:nth-child(7n+5){background:#22d3ee33}.kw span:nth-child(7n+6){background:#6366f144}.kw span:nth-child(7n){background:#c084fc33}
.kw span b{font-family:"JetBrains Mono",monospace;font-weight:500;opacity:.7;margin-left:4px}
.section-title{display:flex;align-items:center;gap:8px;font-weight:800;font-size:15px;margin:.4rem 0 .1rem;color:#fff}
.section-title:after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line2),transparent)}

@media(max-width:720px){
  .block-container{padding:2.2rem 12px 2rem!important}
  .hero,.hero.big{padding:16px}.hero-main{align-items:flex-start}.hero-logo,.hero.big .hero-logo{width:48px;height:48px}
  .eyebrow{letter-spacing:.1em;font-size:9.5px}
  .stats{width:100%}.stat{flex:1;min-width:0}
}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style>
"""
