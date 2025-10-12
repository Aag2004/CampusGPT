from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

# Gemini API Key for embeddings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chroma Cloud credentials
CHROMA_CLOUD_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY")
CHROMA_PROJECT_NAME = os.getenv("CHROMA_DATABASE")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

# Embedding model
embed_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

# 1️⃣ Load documents
docs_path = Path("../rag-chatbot/data/raw") 
all_docs = []

for file in docs_path.iterdir():
    if file.suffix == ".txt":
        loader = TextLoader(str(file), encoding="utf-8")
    elif file.suffix == ".pdf":
        loader = PyPDFLoader(str(file))
    elif file.suffix == ".docx":
        loader = Docx2txtLoader(str(file))
    elif file.suffix in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(str(file))
    else:
        continue
    all_docs.extend(loader.load())

print(all_docs)