import os
from pathlib import Path
from vllm import LLM

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)


class VLLMClient:
    """
    Client for generating embeddings using the VLLM library.

    Usage:
        client = VLLMClient()
        embeddings = client.encode(["text1", "text2"])
    """
    
    def __init__(self):
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "AITeamVN/Vietnamese_Embedding_v2"
        )
        self.llm = LLM(
            model=model_name,
            task="embed",
            enforce_eager=True
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): A list of input strings to embed.

        Returns:
            embeddings (list[list[float]]): A list of embedding vectors (list of floats).
        """
        outputs = self.llm.embed(texts)
        return [output.outputs.embedding for output in outputs]
    