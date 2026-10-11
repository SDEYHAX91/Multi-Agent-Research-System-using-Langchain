# 🔬 ResearchScribe — Multi-Agent Research System

> Four specialized AI agents collaborate — **searching, scraping, writing, and critiquing** — to turn any topic into a polished, source-backed research report.

Built with **LangChain**, **Tavily**, and **Streamlit**, with automatic multi-provider LLM fail-over so a single rate limit never breaks a run.

---

## 📌 Overview

ResearchScribe is a sequential multi-agent pipeline. You enter a topic; each agent handles one stage and hands its output to the next:

| # | Agent | Role | Tool | Default Model |
|---|-------|------|------|---------------|
| 01 | **Search Agent** | Finds recent, reliable web sources | Tavily search | GPT-OSS-120B |
| 02 | **Reader Agent** | Scrapes the most relevant page for deep content | BeautifulSoup scraper | Gemini 3.8 Flash |
| 03 | **Writer** | Drafts a structured report (Introduction, Key Findings, Conclusion, Sources) | LangChain chain | Command A+ |
| 04 | **Critic** | Scores the report out of 10 with strengths, improvements, and a verdict | LangChain chain | Nemotron 3 Super |

## ✨ Features

- **End-to-end research pipeline** — from a single topic to a reviewed report
- **Streamlit web UI** with live per-step progress and timings
- **LLM fail-over** (`llm_failsafe.py`) — on rate limit, quota, auth, or timeout errors, calls are retried on the next model/provider, with per-model cooldowns
- **Role-based model routing** — each agent has its own preferred model and fallbacks
- **Research history** — past reports saved locally and reloadable from the sidebar
- **AI-generated topic suggestions** to get started quickly
- **Download reports** as Markdown (`.md`)
- **CLI mode** for running the pipeline in the terminal

## 🏗️ Architecture

```
Topic ─▶ Search Agent ─▶ Reader Agent ─▶ Writer ─▶ Critic ─▶ Report + Review
          (Tavily)        (Scraper)      (Report)   (Score)
              └──────── llm_failsafe.py: model fail-over per role ────────┘
```

## 📁 Project Structure

```
.
├── app.py            # Streamlit UI (progress, history, report download)
├── pipeline.py       # Orchestrates the 4-step pipeline (CLI entry point)
├── agents.py         # Search/Reader agents, Writer & Critic chains, prompts
├── tools.py          # web_search (Tavily) and scrape_url (BeautifulSoup) tools
├── llm_failsafe.py   # Multi-provider model fail-over and role chains
├── ICON.png          # Project icon
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
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
GROQ_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
COHERE_API_KEY=
NVIDIA_API_KEY=
```

> ⚠️ Never commit your `.env` file.

### Run

**Web app**
```bash
streamlit run app.py
```

**Command line**
```bash
python pipeline.py
```

**Test model providers**
```bash
python llm_failsafe.py
```

## ⚙️ Customizing Models

Edit `ROLE_CHAINS` in `llm_failsafe.py`, or override a role from `.env` using `provider:model` pairs (comma-separated, first = preferred):

```env
LLM_CHAIN_WRITER=cohere:command-a-plus-05-2026,groq:openai/gpt-oss-120b
```

Supported overrides: `LLM_CHAIN_SEARCH`, `LLM_CHAIN_READER`, `LLM_CHAIN_WRITER`, `LLM_CHAIN_CRITIC`.

## 🛠️ Tech Stack

Python · LangChain · Tavily · BeautifulSoup · Streamlit · Groq · OpenRouter · Google Gemini · Cohere · NVIDIA NIM

## 🗺️ Roadmap

- [ ] Let the Reader agent scrape multiple sources
- [ ] Revision loop: Writer improves the report from Critic feedback
- [ ] Export to PDF / DOCX
- [ ] Fail-over for Writer and Critic roles

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

## 📄 License

Distributed under the MIT License. Add a `LICENSE` file before publishing.
