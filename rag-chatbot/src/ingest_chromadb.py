from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

# Hugging Face token (if required for private models)
HF_TOKEN = os.getenv("HF_API_KEY")

# Chroma Cloud credentials
CHROMA_CLOUD_API_KEY = os.getenv("CHROMA_CLOUD_API_KEY")
CHROMA_PROJECT_NAME = os.getenv("CHROMA_DATABASE")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# mistral_embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL")
# embed_model = HuggingFaceEmbeddings(model_name="mistral-embed")


# Use Mistral embeddings (from Hugging Face)
# embed_model = HuggingFaceEndpointEmbeddings(
#     model=mistral_embedding_model 
# )

# Load documents
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

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(all_docs)

# Initialize Chroma Cloud client
client = chromadb.CloudClient(
    api_key=CHROMA_CLOUD_API_KEY,
    tenant=CHROMA_TENANT,
    database="RAG for Academics"
)

# Load or create Chroma collection
collection_name = "ordinance_vectors"

try:
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embed_model,
        client=client
    )
    vectordb.add_documents(chunks)
except Exception:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        collection_name=collection_name,
        client=client
    )

print("Embeddings stored in successfully.")