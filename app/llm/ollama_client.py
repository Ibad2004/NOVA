import os

import ollama
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
       
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        self.temperature = float(
            os.getenv("OLLAMA_TEMPERATURE", "0.2")
        )

        self.max_tokens = int(
            os.getenv("OLLAMA_MAX_TOKENS", "2048")
        )

        self.client = ollama.Client(host=self.base_url)

    def chat(
        self,
        message: str,
        think: bool = False,
    ) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            think=think,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        return response["message"]["content"]