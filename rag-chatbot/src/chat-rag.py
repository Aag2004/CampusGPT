import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1️⃣ Load API key
load_dotenv()
hf_token = os.getenv("HF_API_KEY")

# 2️⃣ Connect to hosted LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=200
)

chat_model = ChatHuggingFace(llm=llm)

# 3️⃣ Load FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("../vectorstore/faiss_store", embeddings, allow_dangerous_deserialization=True)

retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

# 4️⃣ Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5️⃣ Conversational RAG pipeline
conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    verbose=False
)

# 6️⃣ Chat loop
print("🤖 RAG Chatbot started! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! 👋")
        break
    response = conversation.invoke({"question": user_input + "Keep the solution concise."})
    print("Bot:", response["answer"])


# FOR SENDING SYSTEM INSTRUCTION ONLY ONCE

## 1️⃣ Prime the model with instruction once
# system_instruction = "Keep the answers concise."
# conversation.invoke({"question": system_instruction})

# # 2️⃣ Chat loop
# print("🤖 RAG Chatbot started! Type 'quit' to exit.\n")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit", "bye"]:
#         print("Bot: Goodbye! 👋")
#         break
    
#     # Only send user query now
#     response = conversation.invoke({"question": user_input})
#     print("Bot:", response["answer"])
