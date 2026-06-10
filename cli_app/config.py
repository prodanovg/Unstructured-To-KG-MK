import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

def get_llm():
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("OLLAMA_MODEL"),
        temperature=0
    )

def get_embeddings():
    return OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("OLLAMA_EMBED_MODEL")
    )

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")