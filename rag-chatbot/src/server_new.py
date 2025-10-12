from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from chromadb import CloudClient
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_chat_model = os.getenv("MISTRAL_CHAT_MODEL")
mistral_embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL")
hf_token = os.getenv("HF_TOKEN")

chroma_api_key = os.getenv("CHROMA_CLOUD_API_KEY")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")
chroma_collection = os.getenv("CHROMA_COLLECTION")

chat_model = ChatMistralAI(
    model=mistral_chat_model,
    api_key=mistral_api_key,
    max_tokens=300
)

# embeddings = MistralAIEmbeddings(
#     model=mistral_embedding_model,
#     api_key=mistral_api_key
# )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

#400 Bad request fix
def safe_embed_query(text):
    if not text or not text.strip():
        text = " "  # minimal placeholder to avoid empty payload
    return embeddings.embed_documents([text])[0]

# Pass it to retriever when creating it
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
    embedding_function=safe_embed_query  # <- safe wrapper
)

conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    verbose=False
)

history =  ChatMessageHistory()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):

    reply = conversation.invoke({
        "question": request.message + "Keep the answer concise.",
        "chat_history": history.messages
    })

    history.add_user_message(request.message)
    history.add_ai_message(reply['answer'])
    return {"data": reply}

@app.get("/")
def root():
    return {"message": "Welcome to CampusGPT"}