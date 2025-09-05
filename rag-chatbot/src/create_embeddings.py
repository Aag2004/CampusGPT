from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Model for embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents (all .txt files)
docs_path = Path("../rag-chatbot/data/raw")
texts = []
for file in docs_path.glob("*.txt"):
    texts.append(file.read_text(encoding="utf-8"))

# print(docs_path)
# print("docs_path (absolute):", docs_path.resolve())
# print("Exists?:", docs_path.exists())

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for txt in texts:
    chunks.extend(text_splitter.split_text(txt))

# print("Chunks count:", len(chunks))
# print("First chunk sample:", chunks[0] if chunks else "No chunks found")

# Create FAISS vectorstore
vectorstore = FAISS.from_texts(chunks, embed_model)

# Save to disk
vectorstore.save_local("../vectorstore/faiss_store")
print("Vector store created/updated successfully!")