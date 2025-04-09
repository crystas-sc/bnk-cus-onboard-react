import re
# import yfinance as yf
from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver  # persistent memory with thread_id support
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)


# ---- LangGraph State Schema ----
class StateSchema(TypedDict):
    user_id: str
    thread_id: str
    input: str
    output: str
    ticker: str
    history: str  # NEW


# ---- Constants ----
OLLAMA_MODEL = "gemma3:4b"
llm = ChatOllama(model=OLLAMA_MODEL)

# ---- Prompt to detect stock ticker ----
STOCK_DETECT_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Determine whether the user is asking for a stock price.

If the user is asking for a stock price, extract the stock ticker symbol (like AAPL, MSFT, TSLA).
If the user is NOT asking about stock prices, respond with "NONE".

User Input: {input}
Answer:
""")

stock_detect_chain: Runnable = STOCK_DETECT_PROMPT | llm | StrOutputParser()

def get_stock_ticker_from_llm(input_text: str) -> str | None:
    """Use LLM to detect if it's a stock query and return ticker."""
    result = stock_detect_chain.invoke({"input": input_text})
    ticker = result.strip().upper()
    print(f"get_stock_ticker_from_llm result: {result}")
    return None if ticker == "NONE" else ticker


# ---- LangGraph Nodes ----

def router_node(state):
    """Router using LLM to detect stock queries."""
    user_input = state["input"]
    ticker = get_stock_ticker_from_llm(user_input)
    state["ticker"] = ticker
    print(f"Router ticker: {ticker}")
    return {**state, "next_node": "stock" if ticker else "chat"}

def get_next_node(state):
    """LangGraph conditional edge resolver."""
    return state["next_node"]

def stock_price_node(state):
    """Fetch stock price using yfinance."""
    print("stock_price_node")
    ticker = state.get("ticker")
    try:
        # stock = yf.Ticker(ticker)
        # price = stock.info.get("regularMarketPrice", "N/A")
        price = 65
        return {
            **state,
            "output": f"The current price of {ticker} is ${price}"
        }
    except Exception as e:
        return {
            **state,
            "output": f"Couldn't fetch price for {ticker}. Error: {str(e)}"
        }

# def chat_node(state):
#     """Chat node using Ollama."""
#     user_id = state["user_id"]
#     user_input = state["input"]
#     stock_reply = state.get("output", "")
#     combined_input = f"{stock_reply}\nUser: {user_input}"

#     chain = ConversationChain(llm=llm)  # Stateless chain, LangGraph handles memory
#     output = chain.predict(input=combined_input)

#     return {
#         "user_id": user_id,
#         "thread_id": state["thread_id"],
#         "input": user_input,
#         "output": output,
#         "ticker": state.get("ticker", "")
#     }

def chat_node(state):
    user_input = state["input"]
    stock_reply = state.get("output", "")
    prior_history = state.get("history", "")

    combined_input = f"{prior_history}\nUser: {user_input}".strip()

    chain = ConversationChain(llm=llm)  # memoryless
    output = chain.predict(input=combined_input)

    # Store updated history for next call
    updated_history = f"{combined_input}\nAI: {output}".strip()

    return {
        **state,
        "output": output,
        "history": updated_history  # ✅ save back
    }


# ---- Build LangGraph ----

graph_builder = StateGraph(StateSchema)
graph_builder.add_node("router", router_node)
graph_builder.add_node("stock", stock_price_node)
graph_builder.add_node("chat", chat_node)

graph_builder.add_conditional_edges("router", get_next_node, {
    "chat": "chat",
    "stock": "stock"
})
graph_builder.set_entry_point("router")
graph_builder.set_finish_point("chat")
graph_builder.add_edge("stock", "chat")

# ---- Compile graph with persistent memory (supports thread_id) ----
# memory = SqliteSaver.from_path("memory.db")
# memory = SqliteSaver(conn)
graph = graph_builder.compile(checkpointer=memory)

# ---- Example usage with thread_id ----
if __name__ == "__main__":
    # Starting a new thread or continuing one by reusing thread_id
    # input_state = {
    #     "user_id": "user123",
    #     "thread_id": "thread-alpha-1",
    #     "input": "My name is bob",
    #     "output": "",
    #     "ticker": ""
    # }

    # result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    # print("\my name Result:", result["output"])
    # input_state = {
    #     "user_id": "user123",
    #     "thread_id": "thread-alpha-1",
    #     "input": "What is my name?",
    #     "output": "",
    #     "ticker": ""
    # }

    # result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    # print("\what is my name Result:", result["output"])
    input_state = {
        "user_id": "user123",
        "thread_id": "thread-alpha-1",
        "input": "What's the stock price of TSLA?",
        "output": "",
        "ticker": ""
    }

    result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    print("\nFinal Result:", result["output"])
