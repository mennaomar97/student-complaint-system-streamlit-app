# 🧠 Student Complaint Assistant (Streamlit + BERT + OpenAI)

An intelligent Streamlit web app that classifies student complaints and routes them automatically.

- 🧩 Uses a **local fine-tuned BERT model** for institutional category prediction.
- 🤖 Integrates an **AI agent (OpenAI)** to detect _technical_ vs _non-technical_ issues.
- 🎨 Built with a clean Streamlit UI — responsive, gradient hero, and sidebar guidance.
- 🔒 Secure API key handling via `secrets.toml` or environment variables.

---

## 📂 Folder Structure

D:\2025\DEPI\Final Project\streamlit
│
├── app.py ← main app (Streamlit + AI agent + BERT)
├── secrets.toml ← secure OpenAI API key (not tracked in Git)
├── best-bert.ipynb ← your training / fine-tuning notebook
├── guide.JPG ← optional reference image / documentation
├── BERT_BEST/ ← local fine-tuned BERT model folder
│ ├── config.json
│ ├── model.safetensors
│ ├── special_tokens_map.json
│ ├── tokenizer.json
│ ├── tokenizer_config.json
│ └── vocab.txt
└── pycache/ ← auto-generated cache folder

---

## ⚙️ Setup Instructions

### 1️⃣ Prerequisites

- Python 3.10+
- Recommended: create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows

- Install dependencies:
pip install streamlit transformers torch python-dotenv openai

```

- Install dependencies:

```bash
pip install streamlit transformers torch python-dotenv openai
```

### 2️⃣ Configure API Key Securely

- Create a file:
  .streamlit/secrets.toml
- Add:
  [openai]
  api_key = "sk-your-openai-key-here"

### 3️⃣ Run the App

From the project folder:

```bash
streamlit run app.py
```

## The app will:

- Ask the AI agent to detect if the complaint is technical or non-technical.
- For technical complaints → show troubleshooting steps from the agent.
- For non-technical complaints → use your BERT model to predict a department category.
