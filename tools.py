from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from rich import print      # used for printing the results in a formatted way in terminal

load_dotenv()

tavily = TavilyClient()

@tool
def web_search(query: str) -> str:
    """Search the web for recent & reliable information on a topic .
    Returns Titles, URLS & snippets."""
    results = tavily.search(query=query, max_results=5)
    out = []
    for r in results['results']:
        out.append(f"""
        Title: {r['title']}
        URL: {r['url']}
        Snippet: {r['content'][:300]}""")
    return "\n---\n".join(out)

# print(web_search.invoke("Latest updates on India?"))


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:5000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

# print(scrape_url.invoke("https://www.livemint.com/sports/cricket-news/why-india-need-mohammed-shami-at-the-2027-odi-world-cup-11791447118487.html"))
