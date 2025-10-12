from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
import chromadb

# -------------------------------
# 1️⃣ FastAPI Setup
# -------------------------------
app = FastAPI()

# Allow frontend (Vite or React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2️⃣ Load API Keys
# -------------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
chroma_api_key = os.getenv("CHROMA_API_KEY")           
chroma_project = os.getenv("CHROMA_PROJECT_NAME")      

# -------------------------------
# 3️⃣ Initialize LLM (Gemini)
# -------------------------------
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key,
    max_output_tokens=300
)

# -------------------------------
# 4️⃣ Initialize Embeddings (Gemini)
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=gemini_api_key,
)

# -------------------------------
# 5️⃣ Connect to ChromaDB Cloud
# -------------------------------
client = chromadb.Client(
    chroma_api_key=chroma_api_key,
    project_name=chroma_project
)

# ✅ Use a collection inside your project
vectordb = Chroma(
    collection_name="rag_collection",   # you can rename
    embedding_function=embeddings,
    client=client
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 6️⃣ Memory for conversation
# -------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------
# 7️⃣ Conversational RAG pipeline
# -------------------------------
conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    verbose=False
)

# -------------------------------
# 8️⃣ FastAPI Routes
# -------------------------------
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = conversation.invoke({"question": request.message + " Keep the answer concise."})
    return {"answer": reply["answer"]}

@app.get("/")
def root():
    return {"message": "🚀 CampusGPT Backend (Gemini + ChromaDB Cloud) is running!"}



# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# # from src.trial_code import conversation   # import conversation pipeline

# app = FastAPI()


# # Allow frontend (Vite dev runs on http://localhost:5173)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],   # later you can restrict to ["http://localhost:5173"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# # 1️⃣ Load API key
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# # 2️⃣ Connect to Gemini LLM
# chat_model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",   
#     google_api_key=gemini_api_key,
#     max_output_tokens=300
# )

# # 3️⃣ Load FAISS vectorstore (with Hugging Face embeddings for retrieval)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# faiss_index = FAISS.load_local(
#     "../vectorstore/faiss_store",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

# # 4️⃣ Memory for conversation
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # 5️⃣ Conversational RAG pipeline
# conversation = ConversationalRetrievalChain.from_llm(
#     llm=chat_model,
#     retriever=retriever,
#     memory=memory,
#     verbose=False
# )

# class ChatRequest(BaseModel):
#     message: str

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     # Send message to model and get reply
#     reply = conversation.invoke({"question": request.message + " Keep the answer concise."})
#     print(reply)
#     return {"data": reply}

# @app.get("/")
# def root():
#     return {"message": "🚀 CampusGPT Backend is running!"}
# # how much fine to pay if offer rejected?