from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from chromadb import CloudClient

# -------------------- FastAPI Setup --------------------
app = FastAPI()

# Allow frontend (Vite dev runs on http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Load Keys --------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
chroma_api_key = os.getenv("CHROMA_CLOUD_API_KEY")
# chroma_project = os.getenv("CHROMA_PROJECT_NAME")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")
chroma_collection = os.getenv("CHROMA_COLLECTION")

# -------------------- LLM --------------------
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key,
    max_output_tokens=300
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# -------------------- Vector Store --------------------
chroma_client = CloudClient(
    tenant=chroma_tenant,
    database=chroma_database,
    api_key=chroma_api_key
)

# chroma_client = CloudClient()

vectorstore = Chroma(
    client=chroma_client,
    collection_name=chroma_collection,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -------------------- Conversation Chain --------------------
conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    verbose=False
)

# -------------------- Chat History --------------------
history = ChatMessageHistory()

# -------------------- API Models --------------------
class ChatRequest(BaseModel):
    message: str

# -------------------- Endpoints --------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    # Pass chat history explicitly
    reply = conversation.invoke({
        "question": request.message + " Keep the answer concise.",
        "chat_history": history.messages
    })

    # Save the conversation manually
    history.add_user_message(request.message)
    history.add_ai_message(reply["answer"])

    return {"data": reply}

@app.get("/")
def root():
    return {"message": "CampusGPT Backend is running!"}
