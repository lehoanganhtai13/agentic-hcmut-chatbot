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
    Client for reranking between a query and a list of documents using vLLM.

    Usage:
    ```python
        client = VLLMClient()
        scores = client.score(
            query="What is AI?",
            documents=[
                "AI is artificial intelligence.",
                "AI stands for Artificial Intelligence."
            ]
        )
        print(scores)  # [0.95, 0.90]
    ```
    """
    
    def __init__(self):
        model_name = os.getenv(
            "RERANKER_MODEL",
            "AITeamVN/Vietnamese_Reranker"
        )
        self.reranker = LLM(
            model=model_name,
            task="score",
            enforce_eager=True
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """
        Generate scores for a query against a list of documents.

        Args:
            query (str): The query string to score against the documents.
            documents (list[str]): List of document strings to score.

        Returns:
            list[float]: List of scores corresponding to each document.
        """
        outputs = self.reranker.score(query, documents)
        return [output.outputs.score for output in outputs]
    