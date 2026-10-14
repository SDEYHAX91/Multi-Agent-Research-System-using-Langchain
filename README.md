# 🔬 ResearchScribe — Multi-Agent Research System

> Four specialized AI agents collaborate — **searching, scraping, writing, and critiquing** — to turn any topic into a polished, source-backed research report.

Built with **LangChain**, **Tavily**, and **Streamlit**, with automatic multi-provider LLM fail-over so a single rate limit never breaks a run, and a built-in **login / account system** so every user gets their own private research history.

Try: https://multi-agent-research-system-using-28d2.onrender.com/

---

## 📌 Overview

ResearchScribe is a sequential multi-agent pipeline. You sign in, enter a topic; each agent handles one stage and hands its output to the next:

| # | Agent | Role | Tool | Default Model |
|---|-------|------|------|---------------|
| 01 | **Search Agent** | Finds recent, reliable web sources | Tavily search | GPT-OSS-120B |
| 02 | **Reader Agent** | Scrapes the most relevant page for deep content | BeautifulSoup scraper | Gemini 3.8 Flash |
| 03 | **Writer** | Drafts a structured report (Introduction, Key Findings, Conclusion, Sources) | LangChain chain | Command A+ |
| 04 | **Critic** | Scores the report out of 10 with strengths, improvements, and a verdict | LangChain chain | Nemotron 3 Super |

## ✨ Features

- **End-to-end research pipeline** — from a single topic to a reviewed report
- **User accounts** — login and account creation (SQLite + salted PBKDF2-SHA256 password hashing); the app is locked until you sign in
- **Logout with confirmation** — a "Log out?" dialog in the sidebar; logging out clears the whole session so nothing leaks to the next user
- **Private, per-account history** — each user's past reports are saved separately and reloadable from the sidebar
- **Streamlit web UI** with live per-step progress and timings
- **LLM fail-over** (`llm_failsafe.py`) — on rate limit, quota, auth, or timeout errors, calls are retried on the next model/provider, with per-model cooldowns
- **Role-based model routing** — each agent has its own preferred model and fallbacks
- **AI-generated topic suggestions** to get started quickly
- **Download reports** as Markdown (`.md`)

<img width="1916" height="989" alt="image" src="https://github.com/user-attachments/assets/33c43acf-2ffd-4250-81fd-7ce900898472" />

  
## 🏗️ Architecture

```
            ┌─ Login / Create Account (auth.py, users.db) ─┐
            ▼                                              │ Log out
Topic ─▶ Search Agent ─▶ Reader Agent ─▶ Writer ─▶ Critic ─┘
          (Tavily)        (Scraper)      (Report)   (Score)
              └──────── llm_failsafe.py: model fail-over per role ────────┘
```

## 📁 Project Structure

```
.
├── app.py            # Streamlit UI (login gate, progress, history, report download)
├── auth.py           # Authentication: users DB, login/sign-up page, logout dialog
├── pipeline.py       # Orchestrates the 4-step pipeline (CLI entry point)
├── agents.py         # Search/Reader agents, Writer & Critic chains, prompts
├── tools.py          # web_search (Tavily) and scrape_url (BeautifulSoup) tools
├── llm_failsafe.py   # Multi-provider model fail-over and role chains
├── users.db          # SQLite user accounts (auto-created if missing; do NOT commit)
├── history/          # Per-user research history, user_<id>.json (auto-created; do NOT commit)
├── ICON.png          # Project icon
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Streamlit 1.37+ (required for the logout confirmation dialog)
- A [Tavily](https://tavily.com) API key (required for search)
- At least one LLM provider key (see below)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
TAVILY_API_KEY=your_key

# Add the providers you want to use (those without a key are skipped automatically)
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GOOGLE_API_KEY=your_key
COHERE_API_KEY=your_key
NVIDIA_API_KEY=your_key

# Optional: store the accounts database somewhere else (default: users.db next to app.py)
# AUTH_DB_PATH=/var/data/users.db
```

> ⚠️ Never commit your `.env` file, `users.db`, or the `history/` folder. Add them to `.gitignore`:
> ```
> .env
> users.db
> history/
> research_history.json
> ```

### Run

**Web app**
```bash
streamlit run app.py
```

On first launch, open the **✨ Create Account** tab to register, then sign in from the **🔑 Login** tab. `users.db` is created automatically.

**Command line**
```bash
python pipeline.py
```

**Test model providers**
```bash
python llm_failsafe.py
```

## 🔐 Authentication

- Passwords are hashed with **PBKDF2-HMAC-SHA256** (200,000 rounds) using a unique random salt per user; plain-text passwords are never stored.
- Emails are unique and stored lowercase; passwords must be 8–128 characters.
- Login failures return one generic message ("Invalid email or password") so account existence isn't revealed.
- History is stored per account in `history/user_<id>.json`. If you are upgrading from the pre-login version, the old shared `research_history.json` is migrated to the first account (id 1) the first time it loads.
- Sessions live in Streamlit's session state, so a browser refresh signs you out.

## ☁️ Deploying on Render

Create a **Web Service** from your GitHub repo with:

| Setting | Value |
|---------|-------|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |

Then add your API keys (`TAVILY_API_KEY`, `GROQ_API_KEY`, …) under **Environment** — don't commit `.env`.

> ⚠️ **Render's disk is ephemeral.** Without a persistent disk, `users.db` and `history/` are wiped on every deploy or restart, so accounts and saved reports disappear.
> To keep accounts, attach a **Persistent Disk** (paid plans), mount it at e.g. `/var/data`, and set `AUTH_DB_PATH=/var/data/users.db`.
> Note: `history/` is currently stored next to `app.py` and is not covered by `AUTH_DB_PATH`.

## ⚙️ Customizing Models

Edit `ROLE_CHAINS` in `llm_failsafe.py`, or override a role from `.env` using `provider:model` pairs (comma-separated, first = preferred):

```env
LLM_CHAIN_WRITER=cohere:command-a-plus-05-2026,groq:openai/gpt-oss-120b
```

Supported overrides: `LLM_CHAIN_SEARCH`, `LLM_CHAIN_READER`, `LLM_CHAIN_WRITER`, `LLM_CHAIN_CRITIC`.

## 🛠️ Tech Stack

Python · LangChain · Tavily · BeautifulSoup · Streamlit · SQLite · Groq · OpenRouter · Google Gemini · Cohere · NVIDIA NIM

## 🗺️ Roadmap

- [ ] Let the Reader agent scrape multiple sources
- [ ] Revision loop: Writer improves the report from Critic feedback
- [ ] Export to PDF / DOCX
- [ ] Fail-over for Writer and Critic roles
- [ ] Persistent login across browser refreshes (cookie-based session)
- [ ] Configurable history location / hosted database for deployment

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.
