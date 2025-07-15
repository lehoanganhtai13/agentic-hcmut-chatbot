import os
import torch
from vllm import LLM


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
            enforce_eager=True,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
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
    