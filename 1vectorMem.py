__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import yfinance as yf
from typing import TypedDict

from langgraph.graph import StateGraph
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# === LangGraph State Schema ===
class StateSchema(TypedDict):
    user_id: str
    thread_id: str
    input: str
    output: str
    ticker: str

# === Config ===
OLLAMA_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_DIR = "./vector_db"
MAX_MEMORY_TOKENS = 1000  # approximation

# === Initialize models ===
llm = ChatOllama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# === Approximate token count by word count ===
def count_tokens_approx(text: str) -> int:
    return len(text.split())

def token_limited_context(docs, max_tokens=MAX_MEMORY_TOKENS):
    context = []
    total = 0
    for doc in docs:
        tokens = count_tokens_approx(doc.page_content)
        if total + tokens > max_tokens:
            break
        context.append(doc.page_content)
        total += tokens
    return "\n".join(context)

# === Vector Memory ===
def get_vectorstore(user_id: str):
    return Chroma(
        collection_name=f"user_{user_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

def get_user_memory(user_id: str):
    return get_vectorstore(user_id).as_retriever()

# === Stock Ticker Detection Prompt ===
STOCK_DETECT_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Determine whether the user is asking for a stock price.

If they are, return the stock ticker (like AAPL, TSLA). Otherwise, return "NONE".

User Input: {input}
Answer:
""")
stock_detect_chain: Runnable = STOCK_DETECT_PROMPT | llm | StrOutputParser()

def get_stock_ticker_from_llm(input_text: str) -> str | None:
    result = stock_detect_chain.invoke({"input": input_text})
    ticker = result.strip().upper()
    return None if ticker == "NONE" else ticker

# === LangGraph Nodes ===
def router_node(state):
    user_input = state["input"]
    ticker = get_stock_ticker_from_llm(user_input)
    return {**state, "ticker": ticker, "next_node": "stock" if ticker else "chat"}

def get_next_node(state):
    return state["next_node"]

def stock_price_node(state):
    ticker = state.get("ticker")
    try:
        # stock = yf.Ticker(ticker)
        # price = stock.info.get("regularMarketPrice", "N/A")
        price = 45
        return {
            **state,
            "output": f"The current price of {ticker} is ${price}"
        }
    except Exception as e:
        return {
            **state,
            "output": f"Couldn't fetch price for {ticker}. Error: {str(e)}"
        }

def chat_node(state):
    user_id = state["user_id"]
    user_input = state["input"]
    stock_reply = state.get("output", "")

    # === Memory ===
    vectorstore = get_vectorstore(user_id)
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(user_input)
    memory_context = token_limited_context(relevant_docs)

    # === Prompt Construction ===
    prompt = f"{memory_context}\nUser: {user_input}".strip()
    if stock_reply:
        prompt = f"{stock_reply}\n{prompt}"

    # === Predict ===
    chain = ConversationChain(llm=llm)  # stateless
    output = chain.predict(input=prompt)

    # === Save messages to vectorstore ===
    vectorstore.add_texts([f"User: {user_input}", f"AI: {output}"])

    return {
        **state,
        "output": output
    }

# === Build LangGraph ===
graph_builder = StateGraph(StateSchema)
graph_builder.add_node("chat", chat_node)
graph_builder.add_node("stock", stock_price_node)
graph_builder.add_node("router", router_node)
graph_builder.add_conditional_edges("router", get_next_node, {
    "chat": "chat",
    "stock": "stock"
})
graph_builder.set_entry_point("router")
graph_builder.set_finish_point("chat")
graph_builder.add_edge("stock", "chat")

graph = graph_builder.compile()

# === Example Run ===
if __name__ == "__main__":
    input_state = {
        "user_id": "user123",
        "thread_id": "thread-alpha-1",
        "input": "My name is Bob",
        "output": "",
        "ticker": ""
    }

    # result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    # print("\n[Step 1] Result:", result["output"])

    input_state["input"] = "What is my name?"
    result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    print("\n[Step 2] Result:", result["output"])
