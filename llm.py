__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import yfinance as yf
from langgraph.graph import StateGraph
# from langchain_chroma import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
import requests
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb
import json

from langchain_community.vectorstores import Chroma  # Import from the new package

from langchain.memory import ConversationBufferMemory  # You can use this or migrate to the new memory management





class StateSchema(TypedDict):
    user_id: str
    input: str
    output: str
    ticker: str
    asked_for_ticker: bool


OLLAMA_MODEL = "gemma3:4b"
embedding_model = "nomic-embed-text"

# Init LLM and Embedding model
llm = ChatOllama(model=OLLAMA_MODEL)
embeddings = OllamaEmbeddings(model=embedding_model)
CHROMA_DB_DIR = "./vector_db"

vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# Initialize ChromaMemory
memory = ConversationBufferMemory(memory_key="chat_history")

client = chromadb.Client()
collection_name = "user_memory"
collection = client.create_collection(name=collection_name)


chroma_client = Chroma(
    collection_name="user_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

# def get_user_memory(user_id: str):
#     vectorstore = Chroma(
#         collection_name=f"user_{user_id}",
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DB_DIR
#     )
#     retriever = vectorstore.as_retriever()
#     return VectorStoreRetrieverMemory(retriever=retriever)

def get_user_memory(user_id: str):
    """Retrieve or create a persistent memory for the user."""
    try:
        # Query for the user's memory documents
        results = collection.query(
            query_texts=[user_id],  # Query based on user_id
            n_results=1  # Limit to 1 result
        )
        
        if results["documents"]:
            # The result is a list of documents, so get the first document
            memory_data = results["documents"][0]  # Get the first document from the list
            memory_dict = json.loads(memory_data)  # Deserialize it back to a dictionary

            # Create a ConversationBufferMemory instance from the dictionary
            memory = ConversationBufferMemory.from_dict(memory_dict)
            return memory
        
    except Exception as e:
        print(f"Error retrieving memory for user {user_id}: {e}")
    
    # If no memory exists, create a new one
    memory = ConversationBufferMemory(return_messages=True)

    # Save the new memory to Chromadb
    save_user_memory(user_id, memory)
    return memory

def save_user_memory(user_id: str, memory: ConversationBufferMemory):
    """Save the user's memory to ChromaDB."""
    try:
        # Serialize the memory to a JSON string
        memory_messages = memory.get_messages()  # Get the conversation messages as a list
        memory_json = json.dumps(memory_messages)  # Serialize it into a JSON string
        
        # Store the memory document in ChromaDB
        collection.add(
            documents=[memory_json],  # The document should be a JSON string
            metadatas=[{"user_id": user_id}],
            ids=[user_id]
        )
        print(f"Successfully saved memory for user {user_id}.")
        
    except Exception as e:
        print(f"Error saving memory for user {user_id}: {e}")


# def is_stock_query(text: str) -> str | None:
#     """Detects if user is asking for a stock price and extracts the ticker."""
#     match = re.search(r'\b([A-Z]{1,5})\b.*?(stock|price|quote)', text, re.IGNORECASE)
#     if match:
#         return match.group(1).upper()
#     return None

# Prompt to detect and extract a stock ticker
# STOCK_DETECT_PROMPT = PromptTemplate.from_template("""
# You are a helpful assistant. Determine whether the user is asking for a stock price.

# If the user is asking for a stock price, extract the stock ticker symbol (like AAPL, MSFT, TSLA).
# If the user is NOT asking about stock prices, respond with "NONE".

# User Input: {input}
# Answer:
# """)

# stock_detect_chain: Runnable = STOCK_DETECT_PROMPT | llm | StrOutputParser()

# def get_stock_ticker_from_llm(input_text: str) -> str | None:
#     """Use LLM to detect if it's a stock query and return ticker."""
#     result = stock_detect_chain.invoke({"input": input_text})
#     ticker = result.strip().upper()
#     print(f"get_stock_ticker_from_llm result: {result}")
#     return None if ticker == "NONE" else ticker

ADDRESS_DETECT_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Determine whether the user is asking for a company's office address.

If the user is asking for a company's address, extract the stock ticker symbol (like AAPL, POLYCAB).
If the user is NOT asking about an address, respond with "NONE".

User Input: {input}
Answer:
""")

address_detect_chain: Runnable = ADDRESS_DETECT_PROMPT | llm | StrOutputParser()

def get_company_ticker_from_llm(input_text: str) -> str | None:
    """Use LLM to detect if it's a company address query and return ticker."""
    result = address_detect_chain.invoke({"input": input_text})
    ticker = result.strip().upper()
    print(f"get_company_ticker_from_llm result: {result}")
    return None if ticker == "NONE" else ticker

def office_address_node(state):
    ticker = state.get("ticker")
    print(f"office_address_node {ticker}")
    url = f"https://www.nseindia.com/api/corp-info?symbol={ticker}&corpType=compdir&market=debt"
    headers = {
        "accept": "*/*",
        "referer": f"https://www.nseindia.com/get-quotes/equity?symbol={ticker}",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Add realistic browser headers if needed (cookies not strictly necessary unless site blocks)
       "cookie": "AKA_A2=A; bm_mi=3EF78F9109C41A037F2A906FC4F8F964~YAAQDCozamXhDguWAQAAm29ZDBuvDUyPDlr//QLqtDZwPjS/SEhetDA3rR5dYn819Xe4RHYl6+Pnf93s4CV3dcQnwnYYyhbXgtAvMZ2StGF791Y3G8ck+apH4YpZvocYkqNM8B++yFYOVh+j+1Ocj7ffEgQQ33m3RaX6OF6YTGkNtF32ZZum9zByyg/5U0iioe1cMiuN6IKb4MWOwP8/Vv/+k9JwFwguypoj3KTP3x2PRyTmzIZUgxLIrI6LXYhe8C5C9QYVTbHbTPF8miuUJwX2s0ioh/9xxm3XvKJT7lsEo+rI4NP9zdn9l4q4r1pSe/PHtBzXiE7NnkOJ~1; nsit=vSBn-q4byPoV-mY9wzAurwFn; nseappid=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcGkubnNlIiwiYXVkIjoiYXBpLm5zZSIsImlhdCI6MTc0Mzk2MzkxOSwiZXhwIjoxNzQzOTcxMTE5fQ.dE3ZiMZFnOiZuspSg4eFa4Ijxv_PFRERdB9Z0us_31g; bm_sz=7C064E19D357440DDA41400B9C92BEFB~YAAQDCozaj7iDguWAQAAn5ZZDBvXZNYOfbwPWymDQXysiadY3N8KEp0mcNj025HnwQ50zJVyrzrmPPXaxpugxj9nHc2AuOytM25g+JepKV6hBV9TA9SGj8v02ylQltaTxObQS24f5aiXtKz1/RHaZAjX15oNwYVL2flrjrZ4+nZhgue9P01Pk/AoNC4Z9oNKxhv50/ltKo0Npj/QJ8BIvEILSgWNvg80fTgL9x8QvGKl5YJF/wTifsnYQZnIcmiFmVnihp1OEy5n4+7Jaqf0zkrJMq7RchI2BS1kMPJfVvJIR1EnjLhNipGevEj0y5DCMiNDkwUrZF7KMgaUJk9JlutANG+5DioELJw5p2Qk/4FZ3ya+F5xUsZ4GdDGpv40z6tk6aMd536rb1wEG+ublA44l/JQ=~3622456~3490626; _abck=8A6E0D3AD04A74DC420A94837DFA09DB~0~YAAQDCozalXiDguWAQAAdphZDA3QIUVIOncnqEUlGqcl3s8qTl4/7XPrQxKLfy/HwjZHJ69PoaLtl99rVhEC3AdoyuD+8pJ2IMsRyp7P1pElRlah+nUm48j1t7paHyC4NxwR2RNnLEgFleBXXH4MOrZc01T86qdn2YqOg3EC/iHYU3QJpObGrLjkgw20KWjvJ9C0DaKBxTh6L5zvr4BW4NwGSZhi48I73LDqFCgkjm1kWpRg25P7OXqA9mWD/9N1T1gT5qoBgwbYZL50wis95+2Oyn0gWs66faOkMo32iXXaRrKYm51busBTvzB9CFs2TXSaBAqTk1N8+vwTf1el02CzAoniz1qBnn3R2/Evl8vXkiLRx5/98Y6rPduFh8nmIJ9aVgUxbI0rdltI1ziCQdZJLGFWoIOiflbfc6EBSj9lLj6ImMBIrh/F9r8UjypvTS+nYIVNxJYaA2++niSxEn4G7Vd1fRMqx/TWMqlM1rplYSZ533ZwuFTpkPXlMGz7uFO0PKiy0hFzDt9H4SaEDoCTI2+jp1pMiLkr9qS5+g==~-1~-1~-1; ak_bmsc=41E554DC92FF5AF8113F36EC0D47223C~000000000000000000000000000000~YAAQDCozalviDguWAQAAR5lZDBtYSr9jQjOryR//6yzzIIdISVw36TVCNr7s0p0wEvTgROKi4c1jybAwlWI10uv9Zcipst5rSUHWIVSbQXKq86OgWPSQct1qvScA6Wz9C3wAbDbWXfFxAVzANawkBLP0DSJ4TCjdG90oz4rEUHGqeq7imYAgUN+eIpGHTIYB/2uoc4DYn6rtShXTJphDaA23PLoqkPA4/68Q3BbN3SAdbO3BWHOFztGEcoQNcvFM/qaTfUzAIgLGljj0Nwgw6SBBOfS8Pos/VKR1AsOmlumnCTLYVMD2vO0MXk2rsvlVpHkq9qI96L71tS5zVcllqcBN0jpd0aj1Cu3Ruzs8uA+JDNizW6HdBfPKm4RzLIhDs9Q5Rt/swyHZpveprMTxqkqZk+4gJy5C9/bYZ90349ijVFU0wG/cmnEIg2p57BrpkwdVYtRA+lrHdteNG9P2aS2p+5+9bFjdfGs+EMnsS78OqV8d8tDqUHZvOX69o2glSFqp; _ga=GA1.1.1482236096.1743963919; _ga_WM2NSQKJEK=GS1.1.1743963919.1.0.1743963919.0.0.0; _ga_87M7PJ3R97=GS1.1.1743963919.1.1.1743963919.60.0.0; nseQuoteSymbols=[{\"symbol\":\"POLYCAB\",\"identifier\":\"\",\"type\":\"equity\"}]; bm_sv=74C9B6CE5197D835CDB785ED07F660B2~YAAQDCozan/iDguWAQAAcqBZDBvs7OXoe4mz+5HS76GeUWPYgg9QuSUfI068X94WHl0+0o3yNEvF8uaB4y8lj8AOVcpTx6m0DEvvUv4V7nm8RFLrHGgtb/Wt4dxbjWrxYGoP2wn0xJmikprK7RDD8AoRE/W+mMLOyGbKuG/o1aV9Vu3mvnvgqGplSjGjDjdVqtY3NBoGkU9us8y1waiayLBi7bShZvuuyh9FZUvgXOhZy/4gclbDRLuOATM3Ep9zSt3m~1"
        
        ,"User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"office_address_node response {response}")

        response.raise_for_status()
        data = response.json()

        # Extract office address (e.g. from `address`, `corpAddress`, or similar)
        first_item_address = data[0]['address']

        address = first_item_address

        return {
            **state,
            "output": f"The registered office address of {ticker} is:\n{address}"
        }
    except Exception as e:
        return {
            **state,
            "output": f"Couldn't fetch office address for {ticker}. Error: {str(e)}"
        }

def router_node(state):
    user_input = state["input"]
    # ticker = get_stock_ticker_from_llm(user_input)
    ticker = get_company_ticker_from_llm(user_input)

    print(f"Router detected ticker: {ticker}")

    # If ticker found
    if ticker:
        # return {**state, "ticker": ticker, "next_node": "stock"}
        return {**state, "ticker": ticker, "next_node": "address"}

    # No ticker found
    if state.get("asked_for_ticker"):
        # Already asked once; don’t loop again
        return {**state, "next_node": "chat"}
    else:
        # Ask for ticker
        return {**state, "asked_for_ticker": True, "next_node": "ask_ticker"}
    

def ask_ticker_node(state):
    return {
        **state,
        "output": "Sure! Which stock are you asking about? Please provide the ticker symbol (e.g., AAPL, MSFT, TSLA)."
    }

def get_next_node(state):
  """Gets the next node based on the state."""
  return state["next_node"]


def stock_price_node(state):
    """Fetch stock price using yfinance."""
    print("stock_price_node")
    # ticker = is_stock_query(state["input"])
    ticker = state.get("ticker")
    try:
        #stock = yf.Ticker(ticker)
        #price = stock.info.get("regularMarketPrice", "N/A")
        price = 45
        print({
            **state,
            "output": f"The current price of {ticker} is ${price}"
        })

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
    """Chat with memory using Ollama."""
    user_id = state["user_id"]
    user_input = state["input"]
    stock_reply = state.get("output", "")
    # Combine stock response (if any) with user input
    combined_input = f"{stock_reply}\nUser: {user_input}"

    conversation_chain = ConversationChain(llm=llm, memory=memory)


    output = conversation_chain.predict(input="Hello, how are you?")

    # memory = get_user_memory(user_id)

    # def get_session_history():
    #     return memory.chat_memory
    # # chain = ConversationChain(llm=llm, memory=memory)
    # # output = chain.predict(input=combined_input)
    # print(f"get_session_history: {memory.chat_memory.messages}")
    # chain = RunnableWithMessageHistory(
    #     runnable=llm,
    #     memory=memory,
    #     get_session_history=get_session_history
    # )

    # output = chain.invoke({"input": combined_input})

    # # Add user and AI messages to memory
    # memory.chat_memory.add_user_message(user_input)
    # memory.chat_memory.add_ai_message(output)
    # print(f"after get_session_history: {memory.chat_memory.messages}")
    # # Save updated memory to Chroma
    # save_user_memory(user_id, memory)



    return {
        "user_id": user_id,
        "input": user_input,
        "output": output
    }

# LangGraph state definition
state_schema = {
    "user_id": str,
    "input": str,
    "output": str,
}

# Build LangGraph
graph_builder = StateGraph(StateSchema)
graph_builder.add_node("chat", chat_node)
# graph_builder.add_node("stock", stock_price_node)
graph_builder.add_node("ask_ticker", ask_ticker_node)
graph_builder.add_node("router", router_node)
graph_builder.add_node("address", office_address_node)


graph_builder.set_entry_point("router")

# Conditional routing from router
graph_builder.add_conditional_edges("router", get_next_node, {
    # "stock": "stock",
    "address":"address",
    "ask_ticker": "ask_ticker",
    "chat": "chat"

})

# If we asked for ticker, next user input routes to stock price
graph_builder.add_edge("ask_ticker", "router")
#graph_builder.set_finish_point("stock")
graph_builder.set_finish_point("address")


graph = graph_builder.compile()


for step in graph.stream({
    "user_id": "bob",
    "input": "My name is Bober.",
}):
    print("Step result:", step)


for step in graph.stream({
    "user_id": "bob",
    "input": "What is my name",
}):
    print("Step result:", step)