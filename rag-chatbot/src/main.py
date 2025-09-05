from config import get_model

llm = get_model()
response = llm.invoke("Hello, can you explain LangChain in one line?")
print(response)
