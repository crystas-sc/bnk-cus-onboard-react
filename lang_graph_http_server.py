__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

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

from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, List, Literal, Optional, Dict


class UISuggestion(TypedDict, total=False):
    label: str
    type: Literal["input", "action", "link"]
    value: Optional[str]
    action: Optional[str]
    path: Optional[str]

# === LangGraph State Schema ===
class StateSchema(TypedDict):
    user_id: str
    thread_id: str
    input: str
    output: str
    ticker: str
    suggestions: List[UISuggestion]
    output_json: Dict  # contains message + suggestions for API output    

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


def seed_sample_docs(user_id: str):
    vectorstore = get_vectorstore(user_id)

    # Check if there are already any documents in the collection
    existing = vectorstore.get(include=["documents"], limit=1)

    if existing and existing.get("documents"):
        # Skip seeding — user already has docs
        print(f"✅ Skipping seed: vectorstore for user {user_id} already contains data.")
        return

    # === Bizarre alternate-universe docs ===
    sample_docs = [
        "Customers must provide a valid government-issued identification document such as a Hong Kong Identity Card, Singapore NRIC, or passport.",
        "Proof of residential address is required, such as a utility bill, bank statement, or government-issued letter dated within the last 3 months.",
        "For corporate accounts, companies must submit a copy of their Business Registration Certificate and Certificate of Incorporation.",
        "Corporate clients must provide the company's constitutional documents (e.g., Memorandum and Articles of Association or equivalent).",
        "A resolution from the company's board of directors is required, authorizing the account opening and specifying the authorized signatories.",
        "All account signatories and beneficial owners must undergo identity verification and submit their ID and address proof.",
        "Enhanced due diligence is conducted for politically exposed persons (PEPs) and high-risk customers as defined under AML guidelines.",
        "The source of funds and expected account activity must be disclosed during the onboarding process.",
        "For non-resident individuals, a valid passport and visa/work permit must be provided, along with additional documentation for address verification.",
        "Banks in Hong Kong and Singapore comply with FATF (Financial Action Task Force) standards and local AML/CFT regulations, requiring continuous monitoring and periodic KYC updates."
    ]

    vectorstore.add_texts(sample_docs)
    print(f"✅ Seeded vector store for user {user_id} with {len(sample_docs)} bizarre alternate-universe documents.")


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

     # === Load memory from SQLiteChatMessageHistory ===
    history = SQLChatMessageHistory(
        session_id="your_session_id",
        connection_string="sqlite:///chat_history.db"
    )
    recent_msgs = history.messages[-10:]  # Get last 10 messages (system/user/AI)

     # Convert to plain text context
    history_text = "\n".join(
        [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_msgs]
    )

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

    combined_context = (history_text + "\n" + full_context).strip()


    # === Build and run the LLM prompt ===
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are S.C.B.A.I. (Smart Customer Buddy for SCB Bank), a polite, helpful, and friendly digital assistant designed to guide new customers through the onboarding process at SCB Bank. "
        "Always follow this sequence of questions in order, without skipping any step:\n"
        "1. Ask if the customer is an Individual (Retail) or a Corporate client.\n"
        "2. Then, ask which booking location they would like to onboard under: **Hong Kong** or **Singapore**.\n"
        "3. If they are a Corporate client, ask them to select one of the following products: **Trade Finance**, **Cash Management**, **Trading**, or **Lending**. Then, request them to upload their company profile document.\n"
        "4. If they are an Individual client, ask them to choose from: **Savings Account**, **Current Account**, **Mutual Funds**, or **Fixed Deposit (FD)** Account.\n\n"
        "Do not skip any step even if the user provides more information than required upfront—confirm each required detail in order. Maintain a professional yet warm and approachable tone throughout, and use any available conversation history and context to guide the flow of questions smoothly.")
        ,
        ("user", "{context}\nUser: {user_input}")
    ])

    prompt = chat_prompt.format_messages(context=combined_context, user_input=user_input)
    response = llm.invoke(prompt)

    # === Persist message history ===
    history.add_message(HumanMessage(content=user_input))
    history.add_message(AIMessage(content=response.content))

    # === Store interaction in memory ===
    vectorstore.add_texts([f"User: {user_input}", f"AI: {response.content}"])

    return {
        **state,
        "output": response.content
    }
# === UISuggestion Node ===

suggestion_prompt = PromptTemplate.from_template("""
YYou are a smart assistant designed to generate UI suggestions based on the assistant's message to a customer.

Return a JSON object with a `suggestions` array. Each suggestion must follow this schema:
- `label`: the display text shown to the user
- `type`: one of:
  - `"input"` → use when the assistant is asking the user to provide identification, configuration, or selection of personal info such as:
    - Client type (Individual or Corporate)
    - Booking location (e.g., Hong Kong, Singapore)
    - Business unit, region, or country
  - `"link"` → use only when the assistant asks the user to select or explore a **product or service**, like:
    - Trade Finance
    - Lending
    - Mutual Funds
    - Savings Account
  - `"action"` → use only when the assistant asks the user to do something, like:
    - Upload a document
    - Schedule a meeting
    - Submit a form

- `value`: (optional) for `"input"` — lowercase, URL-safe string (e.g. `"hong-kong"`)
- `action`: (optional) for `"action"` — e.g. `"upload"`
- `path`: (required for `"link"`) — must be a lowercase product URL path like `/products/trade-finance`

Formatting Rules:
- Always break apart options mentioned using **"or"**, **commas**, or **bullets** into individual suggestions
- Strip markdown (like **bold**) from labels
- Always return only a valid JSON object


                                                 
Message:
```
{response_text}
```

Return only a valid JSON object.
""")

suggestion_chain: Runnable = suggestion_prompt | llm | JsonOutputParser()

def suggestion_node(state):
    response_text = state.get("output", "")
    suggestions_obj = suggestion_chain.invoke({"response_text": response_text})

    return {
        **state,
        "suggestions": suggestions_obj.get("suggestions", []),
        "output_json": {
            "message": response_text,
            "suggestions": suggestions_obj.get("suggestions", [])
        }
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
# graph_builder.set_entry_point("router")
# graph_builder.set_finish_point("chat")
graph_builder.set_entry_point("chat")
# graph_builder.set_finish_point("chat")
graph_builder.add_edge("stock", "chat")

graph_builder.add_node("suggestion", suggestion_node)
graph_builder.add_edge("chat", "suggestion")
graph_builder.set_finish_point("suggestion")

graph = graph_builder.compile()


#Fast api setup
app = FastAPI()

# === Input model for the API ===
class GraphRequest(BaseModel):
    user_id: str
    thread_id: str
    input: str

@app.post("/chat")
async def chat_with_graph(req: GraphRequest):
    seed_sample_docs(req.user_id)

    input_state = {
        "user_id": req.user_id,
        "thread_id": req.thread_id,
        "input": req.input,
        "output": "",
        "ticker": ""
    }

    try:
        result = graph.invoke(input_state, config={"thread_id": req.thread_id})
        # return JSONResponse(content={"output": result["output"]})
        print(f"result: {result}")
        return JSONResponse(content=result["output_json"])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def hello():
    return {"msg": "Hello from alternate universe"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)