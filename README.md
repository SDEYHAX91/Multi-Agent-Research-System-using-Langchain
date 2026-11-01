# 🔬 ResearchScribe — Multi-Agent Research System

> Four specialized AI agents collaborate — **searching, scraping, writing, and critiquing** — to turn any topic into a polished, source-backed research report.

Built with **LangChain**, **Tavily**, and **Streamlit**, with automatic multi-provider LLM fail-over so a single rate limit never breaks a run, and a built-in **login / account system** so every user gets their own private research history.

Try: https://multi-agent-research-system-using-28d2.onrender.com/

---

## 📌 Overview

ResearchScribe is a multi-agent pipeline. Sign in, enter a topic; each agent handles one stage and hands its output to the next:

| # | Agent | Role | Tool | Default Model |
|---|-------|------|------|---------------|
| 01 | **Search Agent** | Finds recent, reliable web sources | Tavily search | Ministral 14B |
| 02 | **Reader Agent** | Scrapes the most relevant page for deep content | BeautifulSoup scraper | GPT-OSS-120B |
| 03 | **Writer** | Drafts a structured report (Introduction, Key Findings, Conclusion, Sources) | LangChain chain | GPT-OSS-120B |
| 04 | **Critic** | Scores the report out of 10 with strengths, improvements, and a verdict | LangChain chain | Nemotron 3 Super |

## ✨ Features

- **End-to-End Multi-Agent Architecture:** Four specialized agents handle search, reading, report writing, and critique.
- **LLM Failover:** Uses model failover to keep runs working when one provider fails.
- **Automated Web Search:** Finds relevant, current sources for each topic using Tavily API.
- **Deep Content Extraction:** Scrapes selected pages using BeautifulSoup for detailed information.
- **AI-Powered Report Generation:** Automatically generates structured research reports.
- **Quality Review:** Critiques each report and gives it a score.
- **Interactive UI:** Streamlit-based user interface for easy interaction.
- **Pipeline Orchestration:** Seamless coordination of multiple agents

  
## 🏗️ Architecture

```
┌─────────────────────────────┐
│ Input: Research topic       │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐      ┌──────────────────────────┐
│ Search Agent                │─────▶│ Tavily web search        │
│                             │◀─────│ Live web results         │
└──────────────┬──────────────┘      └──────────────────────────┘
               ▼
       Save search results
               │
               ▼
┌─────────────────────────────┐      ┌──────────────────────────┐
│ Reader Agent                │─────▶│ scrape_url               │
│                             │◀─────│ BeautifulSoup extraction │
└──────────────┬──────────────┘      └──────────────────────────┘
               ▼
       Save scraped content
               │_____________________________________
               |                                    |
               ▼                                    ▼
┌─────────────────────────────┐            ┌─────────────────────────────┐ 
│ Writer Chain                │            │ Critic Chain                │     
└──────────────┬──────────────┘            └──────────────┬──────────────┘
               |                                          |
               ▼                                          |
┌───────────────────────────────┐                         |
│ Output                        │<-------------------------
│ (Final report + critic review)│
└───────────────────────────────┘
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
├── .env
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Streamlit 1.37+ (required for the logout confirmation dialog)
- A [Tavily](https://tavily.com) API key (required for search)
- LLM Providers:
      * Mistral API Key
      * Groq API Key
      * Openrouter API Key
      * Ollama API Key
- Optional: You can use at least one LLM Providers if you desire (OpenAI, Anthropic, Gemini, etc).

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```env
TAVILY_API_KEY=your_key

# Add the providers you want to use (those without a key are skipped automatically)
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
MISTRAL_API_KEY=your_key
OLLAMA_API_KEY=your_key

# Account and History stored in Turso Database. For that, add these below to the .env file
TURSO_AUTH_TOKEN
TURSO_DATABASE_URL
```

> ⚠️ Never commit the `.env` file, `users.db`, or the `history/` folder. Add them to `.gitignore`:
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

Python — application logic
Streamlit — web interface
LangChain — agent and model orchestration
Tavily — web search
BeautifulSoup — page scraping
Mistral, Groq, OpenRouter, Ollama — language model providers
Turso / SQLite — accounts and research history storage

## 🗺️ Roadmap

- [ ] Let the Reader agent scrape multiple sources
- [ ] Revision loop: Writer improves the report from Critic feedback
- [ ] Export to PDF / DOCX
- [ ] Fail-over for Writer and Critic roles
- [ ] Persistent login across browser refreshes (cookie-based session)
- [ ] Configurable history location / hosted database for deployment

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.
