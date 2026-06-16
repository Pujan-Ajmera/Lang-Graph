from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


class AgentState(TypedDict):
    query: str
    plan: str
    research: str
    answer: str

def planner(state: AgentState):
    prompt = f"""
    Create a short plan to answer this question.

    Question:
    {state['query']}
    """

    response = llm.invoke(prompt)

    return {
        "plan": response.content
    }


def researcher(state: AgentState):
    prompt = f"""
    Use this plan:

    {state['plan']}

    Generate detailed research notes.
    """

    response = llm.invoke(prompt)

    return {
        "research": response.content
    }


def writer(state: AgentState):
    prompt = f"""
    Question:
    {state['query']}

    Research:
    {state['research']}

    Write a complete answer.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }-

graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.set_entry_point("planner")

graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

app = graph.compile()

result = app.invoke(
    {
        "query": "Explain vector databases in simple terms"
    }
)

print(result["answer"])