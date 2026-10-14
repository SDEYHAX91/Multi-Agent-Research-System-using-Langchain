from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# pyrefly: ignore [missing-import]
from tools import web_search, scrape_url
# pyrefly: ignore [missing-import]
from llm_failsafe import get_model
from dotenv import load_dotenv

load_dotenv()

# search -> GPT-OSS-120B | reader -> Gemini 3.8 Flash | writer -> Command A+ | critic -> Nemotron 3 Super
search_llm = get_model("search", temperature=0.2)
reader_llm = get_model("reader", temperature=0.2)
writer_llm = get_model("writer", temperature=0.2)
critic_llm = get_model("critic", temperature=0.2)

# 1st Agent
def build_search_agent():
    return create_agent(model=search_llm, tools=[web_search])

# 2nd Agent
def build_reader_agent():
    return create_agent(model=reader_llm, tools=[scrape_url])

# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])
writer_chain = writer_prompt | writer_llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to improve:
- ...
- ...

One-line Verdict: 
...""")
])
critic_chain = critic_prompt | critic_llm | StrOutputParser()
