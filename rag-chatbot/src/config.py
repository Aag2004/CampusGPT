import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_model():
    provider = os.getenv("MODEL_PROVIDER", "huggingface")

    if provider == "huggingface":
        return HuggingFaceHub(
            repo_id=os.getenv("HF_MODEL", "google/flan-t5-base"),
            huggingfacehub_api_token=os.getenv("HF_API_KEY")
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    else:
        raise ValueError("Unsupported MODEL_PROVIDER")
