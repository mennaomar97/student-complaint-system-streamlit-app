# ==============================================
# Unified Streamlit App: AI Agent + BERT Classifier
# Layout: Main (left) + Custom Right Panel (no Sidebar)
# ==============================================
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Any, List, Dict, Optional

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# =========================================================
# SECTION 1 — OpenAI Agent
# =========================================================
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT", "25"))

# ---- Regex & helpers for command extraction -------------------
_CMD_LINE_RE = re.compile(
    r"""^\s*(?:\$|pip(?:3)?\b|python(?:3)?\b|python\s+-m\b|conda\b|git\b|npm\b|npx\b|yarn\b|pnpm\b|sudo\b|apt(?:-get)?\b|brew\b|curl\b|wget\b|powershell\b|cmd\s+/c\b|set\s+\w+|export\s+\w+|cd\s+)""",
    re.IGNORECASE | re.VERBOSE,
)
_TRIPLE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)```", re.MULTILINE)
_INLINE_BT_RE = re.compile(r"`([^`]+)`")

def _lines(s: str) -> List[str]:
    return [ln.rstrip("\r") for ln in (s or "").splitlines()]

def _extract_commands_list(code_raw: str) -> List[str]:
    if not code_raw:
        return []
    cands: List[str] = []
    for block in _TRIPLE_BLOCK_RE.findall(code_raw):
        for ln in _lines(block):
            s = ln.strip().lstrip("$").strip()
            if _CMD_LINE_RE.match(s):
                cands.append(s)
    for inline in _INLINE_BT_RE.findall(code_raw):
        s = inline.strip().lstrip("$").strip()
        if _CMD_LINE_RE.match(s):
            cands.append(s)
    seen, dedup = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup

def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (s or "").lower())

def _best_step_idx_for_cmd(cmd: str, steps: List[str]) -> Optional[int]:
    ctoks = set(_tokens(cmd))
    if not ctoks:
        return None
    best_i, best_score = None, 0.0
    for i, step in enumerate(steps):
        stoks = set(_tokens(step))
        if not stoks:
            continue
        inter = len(ctoks & stoks)
        score = inter / max(1, min(len(ctoks), len(stoks)))
        if score > best_score:
            best_i, best_score = i, score
    return best_i if (best_i is not None and best_score >= 0.25) else None

# ---- OpenAI client setup (secrets→.env fallback) --------------
def _get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("openai", {}).get("api_key")
        if key:
            return key
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
    "Follow the JSON schema exactly (keys, types, names).\n"
    "Input text may be in English, Arabic, or dialects (e.g., Egyptian Arabic), or mixed. "
    "Your job is to ALWAYS internally normalize/translate the complaint into English, then reason on that normalized text. "
    "Final JSON output MUST always use English categories, summaries, and keys.\n"
    "\n"
    "Classification rules:\n"
    "- Always determine the root cause of the complaint before assigning a category. "
    "Do not be misled by surface words like 'error', 'system', 'server' unless the cause truly matches a technical category.\n"
    "- If the complaint is NON-TECHNICAL: "
    "  set routing.is_technical=false and steps_to_apply=[]. "
    "  Use one of these categories: 'administrative', 'logistics', 'schedule_issue', 'general_question', 'other_non_technical'.\n"
    "- If the complaint is TECHNICAL: produce BETWEEN 3 AND 6 steps. "
    "  Each step must be ONE clear action. "
    "  If a step requires terminal/CLI commands, put them in step.commands; otherwise, leave step.commands=[]. "
    "  Categories for technical issues (pick the most specific, avoid 'other_technical' unless nothing else fits):\n"
    "    - coding_bug: errors caused by the student's code or logic\n"
    "    - coding_how_to: questions about how to write or implement code/features\n"
    "    - dev_env_tooling: IDE, interpreter, environment setup, dependencies, package installation\n"
    "    - sys_networks: network, server, connectivity, or external API access issues\n"
    "    - data_ml_dl: database queries, data handling, machine learning, deep learning\n"
    "    - theory_concept: conceptual or theoretical programming/CS/ML questions\n"
    "    - other_technical: only if none of the above categories apply\n"
    "\n"
    "Output rules:\n"
    "- Put commands under the matching step, not in solution.code unless a full block is unavoidable.\n"
    "- Keep 'summary' short and student-friendly.\n"
    "- 'verification_checklist' must be concrete actions the student can check.\n"
    "- 'requests_for_more_info' should be [] unless clarifying questions are truly needed.\n"
    "- Use plain ASCII quotes in JSON. Return ONLY the JSON object.\n"
)

# ===== JSON schema the model must follow =====
RESPONSE_SCHEMA = r"""
Return a SINGLE JSON object that matches EXACTLY this schema:

{
  "routing": {
    "is_technical": true,
    "category": "coding_bug | coding_how_to | dev_env_tooling | data_ml_dl | sys_networks | theory_concept | other_technical | non_technical",
    "confidence": 0.0
  },
  "summary": "Short explanation for the student.",
  "steps_to_apply": [
    {
      "text": "One clear action for this step.",
      "commands": ["optional terminal/CLI commands for THIS step (0..N), one per line, no prose"]
    }
  ],
  "verification_checklist": ["bullet checks the student can validate"],
  "requests_for_more_info": ["0..3 questions for the student, or [] if not needed"],
  "solution": {
    "code_language": "bash | python | text | null",
    "code": "OPTIONAL: full code/commands block ONLY IF absolutely needed (prefer step.commands)."
  }
}

Rules:
- Non-technical -> routing.is_technical=false AND steps_to_apply=[]
- Technical -> 3..6 steps, one action per step. If a step needs a command, put it in step.commands.
- No markdown, no backticks around the whole JSON, no commentary—JSON only.
"""

def ai_agent(student_complaint: str) -> dict[str, Any]:
    client = _get_client()
    if not client:
        return {"error": "OpenAI API key not configured. Set st.secrets['openai']['api_key'] or .env: OPENAI_API_KEY", "raw": ""}

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

    # attach extracted commands to closest matching steps
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
        if not cmds:
            return text
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
# SECTION 2 — Streamlit App (No Sidebar)
# =========================================================
st.set_page_config(page_title="Student Complaint Assistant", page_icon="🗂️", layout="wide")

# ---- CSS / Layout ------------------------------------------------
st.markdown("""
<style>
/* Hide the default sidebar completely */
section[data-testid="stSidebar"]{ display:none !important; }

/* Main container width */
section[data-testid="stSidebar"] ~ section div[data-testid="block-container"]{
  max-width:1200px;margin:0 auto;padding-top:.75rem;padding-bottom:1.25rem;
}

html, body { overflow-y: auto !important; background:#0b0f16; }

/* ===== HERO ===== */
.hero{
  width:100%;margin:20px auto 10px auto;padding:28px 32px;border-radius:20px;
  background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;min-height:160px;
  display:flex;flex-direction:column;justify-content:center;align-items:center;
  box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.hero h1{margin:0 0 8px 0;font-weight:800;line-height:1.1;
  font-size:clamp(28px,4vw,44px);text-align:center;width:100%;}
.hero p{margin:0;opacity:.95;line-height:1.5;font-size:clamp(14px,1.6vw,18px);
  text-align:center;width:100%;}

/* ===== INPUT ===== */
.stTextArea textarea{
  min-height:220px !important;font-size:16px;border-radius:12px;
  border:1px solid #263244; background:#0f172a; color:#e5e7eb;
  line-height:1.6; word-break:break-word; white-space:pre-wrap; hyphens:auto;
}
.stTextArea textarea:focus{ outline:none; border:1px solid #3b82f6; }

/* ===== BUTTON ===== */
.stButton>button{
  background:linear-gradient(135deg,#7c3aed,#ec4899);color:#fff;
  padding:10px 20px;border-radius:12px;border:none;font-weight:700;
}
.stButton>button:hover{ filter:brightness(1.06); }

/* ===== MAIN CARDS ===== */
.card{
  background:#0f172a;border:1px solid #1f2937;color:#e5e7eb;
  border-radius:14px;padding:14px 16px; line-height:1.6;
}
.pred-card{
  background:#0f2720;border:1px solid #1f513f;color:#d1fae5;
  border-radius:14px;padding:14px 16px; line-height:1.6;
}
.block-gap{ height: 12px; }

/* ===== Custom Right Panel ===== */
.right-panel .rp-card{
  background:#0f172a; border:1px solid #1f2937; border-radius:16px;
  padding:14px 14px 10px 14px; margin-bottom:12px; box-shadow:0 6px 18px rgba(0,0,0,.18);
}
.right-panel .rp-title{
  display:inline-flex; align-items:center; gap:8px;
  font-weight:800; font-size:14px; color:#e5e7eb; margin-bottom:8px;
}
.right-panel .rp-title .dot{
  width:8px; height:8px; border-radius:50%; background:linear-gradient(135deg,#7c3aed,#ec4899);
}
.right-panel .rp-list{ margin:8px 0 4px 0; padding-left:18px; }
.right-panel .rp-list li{
  color:#cbd5e1; margin:.35rem 0; line-height:1.65; font-size:13.5px;
  list-style-position: outside;
}
.right-panel .badge{
  display:inline-flex; align-items:center; gap:8px;
  font-weight:700; font-size:13px; padding:6px 10px; border-radius:10px;
  border:1px solid #1f2937; margin:6px 0 8px 0;
}
.right-panel .badge.ok{ background:#0f2720; color:#d1fae5; border-color:#1f513f; }
.right-panel .badge.warn{ background:#2a1b15; color:#fecaca; border-color:#7f1d1d; }
.right-panel .example-box{
  background:#0b1320; border:1px solid #1f2937; border-radius:10px;
  padding:10px; color:#cbd5e1; font-size:13.5px; line-height:1.65;
}
/* RTL support inside right panel cards */
.right-panel .rtl{ direction:rtl; text-align:start; }
</style>
""", unsafe_allow_html=True)

# ---- Hero --------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>Student Complaint Assistant</h1>
  <p>Welcome! This tool classifies your complaint to the right department — and if it's a technical issue, our AI agent will guide you with quick, practical steps.</p>
</div>
""", unsafe_allow_html=True)

# ===== BERT classifier from Hugging Face =====
HF_MODEL_NAME = "your_hf_username/student-complaint-bert"  # <-- غيّرها لاسم موديلك على HF

FALLBACK_LABELS = [
    "Certificates_Documents",
    "Courses_Training",
    "Facilities_Logistics",
    "Finance_Admin",
    "IT_Support",
]

LABEL_ALIAS = {
    "LABEL_0": "Certificates_Documents",
    "LABEL_1": "Courses_Training",
    "LABEL_2": "Facilities_Logistics",
    "LABEL_3": "Finance_Admin",
    "LABEL_4": "IT_Support",
}

@st.cache_resource(show_spinner=False)
def load_model():
    tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    mdl.eval().to("cpu")

    id2label = getattr(mdl.config, "id2label", None)
    labels = None
    if isinstance(id2label, dict) and len(id2label):
        labels = [id2label.get(str(i), id2label.get(i)) for i in range(len(id2label))]
    if not labels:
        labels = FALLBACK_LABELS

    return tok, mdl, labels

tokenizer, model, LABELS = load_model()


def classify_top1(text: str):
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        conf, idx = torch.max(probs, dim=-1)
        label = LABELS[int(idx)]
        return label, float(conf.item())

# =========================================================
# Layout: Left (main app) + Right (helper panel)
# =========================================================
left_col, right_col = st.columns([1.9, 1], vertical_alignment="top")

# ---------- Left column: Main app ----------
with left_col:
    st.markdown('### 📝 Enter the complaint')
    with st.form("clf_form", clear_on_submit=False):
        text = st.text_area(
            " ",
            placeholder="e.g., I need to issue my graduation certificate / I have a problem paying the fees ...",
            label_visibility="collapsed",
            key="complaint_text",
        )
        submitted = st.form_submit_button("Classify", type="primary")


    pred_box = st.container()

    if submitted:
        if not text.strip():
            st.warning("Please enter the complaint text first.")
        else:
            with st.spinner("Asking the AI agent..."):
                agent_result = ai_agent(text)
                agent_view = for_frontend(agent_result)

            if agent_view.get("status") == "error":
                st.error(agent_view.get("message", "Agent error"))
            else:
                st.toast("AI agent finished", icon="✅")
                st.session_state["agent_view"] = agent_view
                st.session_state["agent_result_raw"] = agent_result

            st.markdown("### 🤖 AI Agent Result")
            if agent_view.get("is_technical"):
                st.markdown(f"**Category:** `{agent_view.get('category')}`")
                
                if agent_view.get("summary"):
                    st.markdown(f'<div class="card">{agent_view["summary"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="block-gap"></div>', unsafe_allow_html=True)
                
                if agent_view.get("steps"):
                    st.markdown("**Steps to try:**")
                    for i, step in enumerate(agent_view["steps"], start=1):
                        st.markdown(f"{i}. {step}")
                
                if agent_view.get("code"):
                    st.markdown("**Suggested code/commands:**")
                    st.code(agent_view["code"], language=agent_view.get("code_language") or "text")


            else:
                st.info("This looks non-technical.")

                with st.spinner("Classifying (BERT)..."):
                    top_label, top_conf = classify_top1(text)
                display_label = LABEL_ALIAS.get(top_label, top_label)

                st.markdown('### ✅ Prediction')
                st.markdown(
                    f'<div class="pred-card">Predicted category: <b>{display_label}</b> | Confidence: <b>{top_conf*100:.1f}%</b></div>',
                    unsafe_allow_html=True
                )
                st.progress(top_conf)
    else:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ---------- Right column: Custom helper panel ----------
with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    lang_en = st.toggle("English Tips", value=False, help="Switch tips language")

    if lang_en:
        # ===== English version =====
        st.markdown("""
        <style>
        /* English: left aligned */
        .right-panel { direction: ltr; text-align: left; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rp-card">
        <div class="rp-title"><span class="dot"></span> Student Tips (EN)</div>
        <ul class="rp-list">
            <li>Be <b>specific</b>: system/place, date/time, and any <b>attempts</b>.</li>
            <li>One topic per ticket (submit separate tickets if needed).</li>
            <li><b>No sensitive data</b> (passwords, card numbers).</li>
            <li>Keep it <b>short & clear</b> (1–4 sentences).</li>
        </ul>

        <div class="badge ok">✅ Good example</div>
        <div class="example-box">
            I need the English graduation certificate from Students Affairs. 
            I submitted a request on 10/10 but got no reply. 
            What is the expected processing time?
        </div>

        <br>

        <div class="badge warn">⚠️ Not helpful</div>
        <div class="example-box" style="background:#1a1211; border-color:#7f1d1d; color:#fca5a5;">
            Everything is broken. (too vague)
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        # Notes (English)
        st.markdown("""
        <div class="rp-card">
        <div class="rp-title"><span class="dot"></span> Notes</div>
        <ul class="rp-list">
            <li>The classification is an <b>automatic suggestion</b> and may be adjusted internally.</li>
            <li>If the issue is <b>purely technical</b> (Email/Network/VPN), please mention the type of service, device, or browser.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ===== Arabic version =====
        st.markdown("""
        <style>
        /* Arabic: right aligned */
        .right-panel { direction: rtl; text-align: right; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rp-card rtl">
        <div class="rp-title"><span class="dot"></span> إرشادات للطالب</div>
        <ul class="rp-list">
            <li><b>اكتب المشكلة بوضوح</b>: النظام/المكان + التاريخ/الوقت + أي <b>محاولات</b> قمت بها.</li>
            <li><b>موضوع واحد لكل شكوى</b> (لو في أكثر من موضوع، ابعت شكاوى منفصلة).</li>
            <li><b>لا تكتب بيانات حساسة</b> (كلمات سر/أرقام بطاقات).</li>
            <li><b>اختصر بوضوح</b> (1–4 جمل عادةً كافية).</li>
        </ul>

        <div class="badge ok">✅ مثال جيد</div>
        <div class="example-box">
            أحتاج استخراج شهادة التخرج باللغة الإنجليزية من شؤون الطلاب. 
            قدّمت طلبًا يوم 10/10 ولم أتلقَّ ردًا. 
            ما المدة المتوقعة؟
        </div>

        <br>

        <div class="badge warn">⚠️ مثال غير مناسب</div>
        <div class="example-box" style="background:#1a1211; border-color:#7f1d1d; color:#fca5a5;">
            الموقع سيئ وكل شيء لا يعمل — وصف عام وغير محدد.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="rp-card rtl">
        <div class="rp-title"><span class="dot"></span> ملاحظات</div>
        <ul class="rp-list">
            <li>التصنيف <b>اقتراح آلي</b> وقد يتمّ تعديله داخليًا.</li>
            <li>لو المشكلة <b>تقنية بحتة</b> (إيميل/شبكة/VPN)، اذكر نوع الخدمة والجهاز/المتصفح.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
