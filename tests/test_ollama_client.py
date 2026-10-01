from app.llm.ollama_client import OllamaClient


client = OllamaClient()

response = client.chat(
    "Hello NOVA. Give me a very short introduction in 2 lines.",
    think=False,
)

print("\nNOVA:", response)