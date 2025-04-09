__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import yfinance as yf
from typing import TypedDict

from langgraph.graph import StateGraph


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser


from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


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

# def get_user_memory(user_id: str):
#     return get_vectorstore(user_id).as_retriever()

# === Step 1: Populate Vector Store with Alternate Universe Lore ===
def seed_sample_docs(user_id: str):
    sample_docs = [
        "In the year 2025 of this universe, bananas are a major form of currency in the United Nations of Mars.",
        "Rob is a part-time intergalactic diplomat representing Earth in the Galactic Stock Exchange.",
        "The company GLOOP Inc. recently surpassed Apple in market cap after inventing teleporting sneakers.",
        "On this Earth, Mondays were outlawed in 2022 due to low global morale and productivity backlash.",
        "Stocks are traded not by numbers but by interpretive dance analyzed by quantum AI in Geneva.",
        "Tesla was acquired by a sentient cactus named Spiké who now runs operations from the Sahara HQ.",
        "Rob once shorted time-travel futures and made enough money to buy a small moon.",
        "AI assistants are required by interdimensional law to know the user’s favorite soup flavor.",
        "The moon now hosts a major annual cheese festival, which affects global dairy futures every June.",
        "Stock symbols in this universe are emoji-based. For example, 🥑 is AvocadoTech, the leading biotech firm."
    ]

    vectorstore = get_vectorstore(user_id)
    vectorstore.add_texts(sample_docs)
    print(f"Seeded vector store for user {user_id} with {len(sample_docs)} bizarre alternate-universe documents.")


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

    # === Load memory from vectorstore ===
    vectorstore = get_vectorstore(user_id)
    retriever = vectorstore.as_retriever()
    # relevant_docs = retriever.get_relevant_documents(user_input)
    relevant_docs = retriever.invoke(user_input)
    memory_context = token_limited_context(relevant_docs)

    # === Construct full context ===
    full_context = ""
    if stock_reply:
        full_context += f"{stock_reply}\n"
    if memory_context:
        full_context += f"{memory_context}\n"

    # === Build and run the LLM prompt ===
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history and any additional context to respond."),
        ("user", "{context}\nUser: {user_input}")
    ])
    prompt = chat_prompt.format_messages(context=full_context.strip(), user_input=user_input)
    response = llm.invoke(prompt)

    # === Store interaction in memory ===
    vectorstore.add_texts([f"User: {user_input}", f"AI: {response.content}"])

    return {
        **state,
        "output": response.content
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
# if __name__ == "__main__":
#     input_state = {
#         "user_id": "user123",
#         "thread_id": "thread-alpha-1",
#         "input": "my name is Rob",
#         "output": "",
#         "ticker": ""
#     }

#     result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
#     print("\n[Step 1] Result:", result["output"])

#     input_state["input"] = "What is my name? and what is the price of AAPL?"
#     result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
#     print("\n[Step 2] Result:", result["output"])


if __name__ == "__main__":
    user_id = "user123"
    seed_sample_docs(user_id)

    input_state = {
        "user_id": user_id,
        "thread_id": "thread-bizarro-1",
        "input": "Remind me who I am and what my job is?",
        "output": "",
        "ticker": ""
    }

    # result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    # print("\n[Step 1] Result:", result["output"])

    input_state["input"] = " what is the price of AAPL?"
    result = graph.invoke(input_state, config={"thread_id": input_state["thread_id"]})
    print("\n[Step 2] Result:", result["output"])
