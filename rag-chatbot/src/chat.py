import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 1️⃣ Load API key from .env
load_dotenv()
hf_token = os.getenv("HF_API_KEY")

# 2️⃣ Connect to the hosted model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=100  # Limit output length
)

# 3️⃣ Wrap model for chat
chat_model = ChatHuggingFace(llm=llm)

# 4️⃣ Set up memory to remember conversation
memory = ConversationBufferMemory()

# 5️⃣ Create a conversational pipeline
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=False
)

# 6️⃣ Start chat loop
print("🤖 Chatbot started! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! 👋")
        break
    response = conversation.predict(input=user_input + "keep the solution concise.")
    print("Bot:", response)
