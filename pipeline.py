# pyrefly: ignore [missing-import]
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:

    state = {}
    
    # 1. search agent working
    print("\n"+" ="*50)
    print("Step 1 - search agent is working ...")
    print("="*50)
    
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_result"] = search_result["messages"][1].content
    print("\n search result " ,state["search_result"])

    # 2. reader agent working
    print("\n"+" ="*50)
    print("Step 2 - reader agent is scraping top resources ...")
    print(" ="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages" : [("user", 
            f"Based on the following search results about '{topic}'. "
            f"Pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\n scraped content : ", state['scraped_content'])
    
    # 3. writer agent working
    print("\n" + " ="*50)
    print("Step 3 - writer agent is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS: {state['search_result']} \n\n"
        f"DETAILED SCRAPED CONTENT: {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research": research_combined
    })
    
    print("\n Final Report : ", state['report'])

    # 4. critic report  
    print("\n"+" ="*50)
    print("Step 4 - critic agent is reviewing the report")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report" : state["report"]
    })

    print("\n Critic Report \n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)    
