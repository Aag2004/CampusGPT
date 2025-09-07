from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
import os

# Embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Path setup
docs_path = Path("../rag-chatbot/data/raw")
vectorstore_path = "../vectorstore/faiss_store"

# Load documents
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

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_docs)

# If FAISS store already exists → load it, otherwise create new
if os.path.exists(vectorstore_path):
    vectorstore = FAISS.load_local(vectorstore_path, embed_model, allow_dangerous_deserialization=True)
    vectorstore.add_documents(chunks)  # ✅ add new docs
else:
    vectorstore = FAISS.from_documents(chunks, embed_model)

# Save updated store
vectorstore.save_local(vectorstore_path)
print("Vector store updated successfully!")
