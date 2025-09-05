import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load .env
load_dotenv()
hf_token = os.getenv("HF_API_KEY")

# Use a model that supports text-generation (chat models only work with specific repos)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Free instruct model
    huggingfacehub_api_token=hf_token,            # Correct arg
)

# Wrap in Chat interface
model = ChatHuggingFace(llm=llm)

# Call the model
result = model.invoke("What is the capital of India?")
print(result)