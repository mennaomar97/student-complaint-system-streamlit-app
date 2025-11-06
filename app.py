# ==============================================
# Unified Streamlit App: AI Agent + BERT Classifier
# ==============================================
from __future__ import annotations
import os, io, json, re, sys, time, contextlib
from pathlib import Path
from typing import Any, List, Dict, Optional

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# =========================================================
# SECTION 1 — OpenAI Agent (merged from complaint_agent.py)
# =========================================================
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT", "25"))

# Utility regex and helper functions for command extraction
_CMD_LINE_RE = re.compile(
    r"""^\s*(?:\$|pip(?:3)?\b|python(?:3)?\b|python\s+-m\b|conda\b|git\b|npm\b|npx\b|yarn\b|pnpm\b|sudo\b|apt(?:-get)?\b|brew\b|curl\b|wget\b|powershell\b|cmd\s+/c\b|set\s+\w+|export\s+\w+|cd\s+)""",
    re.IGNORECASE | re.VERBOSE,
)
_TRIPLE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)```", re.MULTILINE)
_INLINE_BT_RE = re.compile(r"`([^`]+)`")

def _lines(s: str) -> List[str]: return [ln.rstrip("\r") for ln in (s or "").splitlines()]
def _extract_commands_list(code_raw: str) -> List[str]:
    if not code_raw: return []
    cands: List[str] = []
    for block in _TRIPLE_BLOCK_RE.findall(code_raw):
        for ln in _lines(block):
            s = ln.strip().lstrip("$").strip()
            if _CMD_LINE_RE.match(s): cands.append(s)
    for inline in _INLINE_BT_RE.findall(code_raw):
        s = inline.strip().lstrip("$").strip()
        if _CMD_LINE_RE.match(s): cands.append(s)
    seen, dedup = set(), []
    for c in cands:
        if c not in seen: seen.add(c); dedup.append(c)
    return dedup

def _tokens(s: str) -> List[str]: return re.findall(r"[a-z0-9_]+", (s or "").lower())
def _best_step_idx_for_cmd(cmd: str, steps: List[str]) -> Optional[int]:
    ctoks = set(_tokens(cmd)); 
    if not ctoks: return None
    best_i, best_score = None, 0.0
    for i, step in enumerate(steps):
        stoks = set(_tokens(step))
        if not stoks: continue
        inter = len(ctoks & stoks)
        score = inter / max(1, min(len(ctoks), len(stoks)))
        if score > best_score: best_i, best_score = i, score
    return best_i if (best_i is not None and best_score >= 0.25) else None

# ============= OpenAI client setup (lazy) =============
def _get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("openai", {}).get("api_key")
        if key: return key
    except Exception:
        pass
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def _get_client() -> Optional[OpenAI]:
    key = _get_api_key()
    return OpenAI(api_key=key) if key else None

SYSTEM_PROMPT = (
    "You are an AI teaching assistant for a student helpdesk. "
    "Return STRICT JSON ONLY (no markdown, no extra text). "
    "Follow the JSON schema exactly (keys, types, names). "
    "Input text may be in English, Arabic, or dialects. "
    "Always reason in English internally.\n\n"
    "If non-technical -> is_technical=false & steps_to_apply=[]. "
    "If technical -> give 3–6 clear steps. "
)
RESPONSE_SCHEMA = """{
  "routing": {"is_technical": true,"category": "coding_bug | sys_networks | data_ml_dl | theory_concept | other_technical | non_technical","confidence": 0.0},
  "summary": "Short explanation",
  "steps_to_apply": [{"text": "Step text","commands": ["optional commands"]}],
  "verification_checklist": [],
  "requests_for_more_info": [],
  "solution": {"code_language": "bash | python | text | null","code": "optional code"}
}"""

def ai_agent(student_complaint: str) -> dict[str, Any]:
    client = _get_client()
    if not client:
        return {"error": "OpenAI API key not configured.", "raw": ""}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{RESPONSE_SCHEMA}\n\nStudent complaint:\n{student_complaint}"},
            ],
            timeout=LLM_TIMEOUT_S,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        return parsed
    except OpenAIError as e:
        return {"error": f"OpenAI API error: {e}", "raw": ""}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "raw": ""}

def for_frontend(agent_result: dict[str, Any]) -> dict[str, Any]:
    if "error" in agent_result:
        return {"status": "error", "message": agent_result["error"]}
    routing = agent_result.get("routing") or {}
    is_technical = bool(routing.get("is_technical", True))
    category = routing.get("category")
    summary = agent_result.get("summary", "")
    verify = agent_result.get("verification_checklist", [])
    steps_in = agent_result.get("steps_to_apply", [])
    sol = agent_result.get("solution", {})
    code_raw = (sol.get("code") or "").strip()

    if code_raw:
        cmds = _extract_commands_list(code_raw)
        if cmds:
            steps_texts = [s.get("text", "") for s in steps_in]
            for cmd in cmds:
                idx = _best_step_idx_for_cmd(cmd, steps_texts)
                if idx is not None:
                    steps_in[idx].setdefault("commands", []).append(cmd)
                else:
                    steps_in.append({"text": "Run the following commands/code:", "commands": [cmd]})

    def _merge(step):
        text = step.get("text", "").strip()
        cmds = [c.strip() for c in step.get("commands", []) if c.strip()]
        if not cmds: return text
        return text + " by running " + "; ".join(f"`{c}`" for c in cmds)

    steps_out = [_merge(s) for s in steps_in if s.get("text")]
    ui = {
        "status": "ok",
        "is_technical": is_technical,
        "category": category,
        "summary": summary,
        "steps": steps_out,
        "verify": verify,
        "code_language": sol.get("code_language"),
        "code": code_raw,
        "ticket_prefill": f"[AI Routing] type={'technical' if is_technical else 'non-technical'}; category={category}\n[Summary]\n{summary}\n[Steps]\n" + "\n".join(f"- {s}" for s in steps_out),
    }
    if not is_technical:
        ui.update({"summary": None, "steps": [], "verify": [], "code_language": None, "code": None})
    return ui


# =========================================================
# SECTION 2 — Streamlit App (merged original UI from app.py)
# =========================================================
st.set_page_config(page_title="Student Complaint Assistant", page_icon="🗂️", layout="wide")

# CSS / Layout
st.markdown("""
<style>
html,body {overflow-y:scroll!important;}
section[data-testid="stSidebar"] ~ section div[data-testid="block-container"]{
 max-width:1100px;margin:0 auto;padding-top:0.75rem;padding-bottom:1.25rem;}
.hero{width:100%;margin:20px auto 10px auto;padding:28px 32px;border-radius:20px;
background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;min-height:160px;
display:flex;flex-direction:column;justify-content:center;align-items:center;
box-shadow:0 10px 30px rgba(0,0,0,.25);}
.hero h1{margin:0 0 8px 0;font-weight:800;line-height:1.1;
font-size:clamp(28px,4vw,44px);text-align:center;width:100%;}
.hero p{margin:0;opacity:.95;line-height:1.5;font-size:clamp(14px,1.6vw,18px);
text-align:center;width:100%;}
.stTextArea textarea{min-height:260px!important;font-size:16px;
border-radius:12px;border:1px solid #3b3b3b;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;
padding:10px 20px;border-radius:12px;border:none;font-weight:700;}
.stButton>button:hover{filter:brightness(1.05);}
.pred-card{background:#0f2720;border:1px solid #1f513f;color:#d1fae5;
border-radius:14px;padding:14px 16px;}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
  <h1>Student Complaint Assistant</h1>
  <p>Welcome! This tool classifies your complaint to the right department — and if it's a technical issue, our AI agent will guide you with quick, practical steps.</p>
</div>
""", unsafe_allow_html=True)

# Load BERT model
MODEL_DIR = Path(os.getenv("CLASSIFIER_MODEL_DIR", "./BERT_BEST"))
FALLBACK_LABELS = [
    "Certificates_Documents",
    "Courses_Training",
    "Facilities_Logistics",
    "Finance_Admin",
    "IT_Support",
]

# In config, save generic names
LABEL_ALIAS = {
    "LABEL_0": "Certificates_Documents",
    "LABEL_1": "Courses_Training",
    "LABEL_2": "Facilities_Logistics",
    "LABEL_3": "Finance_Admin",
    "LABEL_4": "IT_Support",
}

@st.cache_resource(show_spinner=False)
def load_model(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval().to("cpu")
    id2label = getattr(mdl.config, "id2label", None)
    labels = None
    if isinstance(id2label, dict) and len(id2label):
        labels = [id2label.get(str(i), id2label.get(i)) for i in range(len(id2label))]
    if not labels:
        labels = FALLBACK_LABELS
    return tok, mdl, labels
tokenizer, model, LABELS = load_model(MODEL_DIR)

def classify_top1(text: str):
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        conf, idx = torch.max(probs, dim=-1)
        label = LABELS[int(idx)]
        return label, float(conf.item())

# Form
st.markdown('### 📝 Enter the complaint')
text = st.text_area(" ", placeholder="e.g., I need to issue my graduation certificate / I have a problem paying the fees ...", label_visibility="collapsed")
col_btn, _ = st.columns([1,3])
with col_btn:
    run = st.button("Classify", type="primary")
st.caption("The model will predict the most suitable category among the five classes.")

pred_box = st.container()

# === Agent first, then BERT if non-technical ===
with pred_box:
    if run:
        if not text.strip():
            st.warning("Please enter the complaint text first.")
        else:
            with st.spinner("Asking the AI agent..."):
                agent_result = ai_agent(text)
                agent_view = for_frontend(agent_result)

            st.markdown("### 🤖 AI Agent Result")
            if agent_view.get("status") == "error":
                st.error(agent_view.get("message", "Agent error"))
            elif agent_view.get("is_technical"):
                st.markdown(f"**Category:** `{agent_view.get('category')}`")
                if agent_view.get("summary"): st.write(agent_view["summary"])
                if agent_view.get("steps"):
                    st.markdown("**Steps to try:**")
                    for i, step in enumerate(agent_view["steps"], start=1):
                        st.markdown(f"{i}. {step}")
                if agent_view.get("code"):
                    st.code(agent_view["code"], language=agent_view.get("code_language") or "text")
            else:
                st.info("This looks non-technical. You can open a ticket with the following pre-filled text:")
                st.text_area("Ticket text", value=agent_view.get("ticket_prefill", ""), height=160)
                with st.spinner("Classifying..."):
                    top_label, top_conf = classify_top1(text)
                display_label = LABEL_ALIAS.get(top_label, top_label)
                st.markdown('### ✅ Prediction')
                st.markdown(f'<div class="pred-card">Predicted category: <b>{display_label}</b> | Confidence: <b>{top_conf*100:.1f}%</b></div>', unsafe_allow_html=True)
                st.progress(top_conf)
    else:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.divider()

    lang_en = st.toggle("English Tips", value=False)

    if lang_en:
        st.markdown("### 🧭 Student Tips (EN)")
        st.markdown(
            """
- Be **specific**: problem, place/system, date/time, and any **attempts** you made.
- One topic per complaint (submit separate tickets if needed).
- **No sensitive data** (passwords, card numbers).
- Keep it **short & clear** (1–4 sentences).
            """
        )
        with st.expander("✅ Good example"):
            st.markdown(
                """ 
*I need the English graduation certificate from Students Affairs. I submitted a request on 10/10 but got no reply. What is the expected processing time?*
                """
            )
        with st.expander("⚠️ Not helpful"):
            st.markdown("*Everything is broken.* – Too vague; no place/time/expected action.")
    else:
        st.markdown("### 🧭 إرشادات للطالب")
        st.markdown(
            """
- **اكتب المشكلة بوضوح**: النظام/المكان + التاريخ/الوقت + أي **محاولات** قمت بها.
- **موضوع واحد لكل شكوى** (لو في أكثر من موضوع، ابعتي شكاوى منفصلة).
- **لا تكتبي بيانات حساسة** (كلمات سر/أرقام بطاقات).
- **اختصري بوضوح** (1–4 جمل عادةً كافية).
            """
        )
        with st.expander("✅ مثال جيد"):
            st.markdown(
                """
 *أحتاج استخراج شهادة التخرج باللغة الإنجليزية من شؤون الطلاب. قدّمت طلبًا يوم 10/10 ولم أتلقَّ ردًا. ما المدة المتوقعة؟*
                """
            )
        with st.expander("⚠️ مثال غير مناسب"):
            st.markdown("*الموقع سيئ وكل شيء لا يعمل* — وصف عام وغير محدد.")

    st.divider()
    st.markdown(
        """
**📌 ملاحظات:**  
- التصنيف **اقتراح آلي** لمساعدة الجهة المختصة، وقد يتم تعديله داخليًا.  
- لو المشكلة **تقنية بحتة** (إيميل/شبكة/VPN)، اذكري نوع الخدمة والجهاز/المتصفح.
        """
    )
