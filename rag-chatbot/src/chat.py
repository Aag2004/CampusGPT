import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1️⃣ Load API key from .env
load_dotenv()
hf_token = os.getenv("HF_API_KEY")

if not hf_token:
    raise ValueError("⚠️ HF_API_KEY not found in .env file!")

# 2️⃣ Connect to the hosted HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=150,   # controls output length
)

# 3️⃣ Wrap model for chat
chat_model = ChatHuggingFace(llm=llm)

# 4️⃣ Set up memory (remembers conversation history)
memory = ConversationBufferMemory()

# 5️⃣ Create a conversational pipeline
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=False
)
