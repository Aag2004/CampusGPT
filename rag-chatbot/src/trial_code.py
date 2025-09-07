import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1 Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# 2 Connect to Gemini LLM
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   
    google_api_key=gemini_api_key,
    max_output_tokens=300
)

# 3 Load FAISS vectorstore (with Hugging Face embeddings for retrieval)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local(
    "../vectorstore/faiss_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

# 4 Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5 Conversational RAG pipeline
conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    verbose=False
)

# 6 Chat loop
print("RAG Chatbot started! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    response = conversation.invoke({"question": user_input + " Keep the solution concise."})
    print("Bot:", response["answer"])


# 🔹 FOR SENDING SYSTEM INSTRUCTION ONLY ONCE

# 1️⃣ Prime the model with instruction once
# system_instruction = "Keep the answers concise."
# conversation.invoke({"question": system_instructi on})

# 2️⃣ Chat loop
# print("🤖 RAG Chatbot (Gemini) started! Type 'quit' to exit.\n")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit", "bye"]:
#         print("Bot: Goodbye! 👋")
#         break
#     response = conversation.invoke({"question": user_input})
#     print("Bot:", response["answer"])