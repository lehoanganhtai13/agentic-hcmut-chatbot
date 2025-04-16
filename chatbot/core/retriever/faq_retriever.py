from typing import Dict, List

from chatbot.core.retriever.base_class import (
    FAQNode,
    RetrievedFAQ,
    FAQRetrievalResult
)
from chatbot.core.retriever.base_retriever import BaseHybridRetriever
from chatbot.core.model_clients import BM25Client, EmbedderCore
from chatbot.utils.database_clients import VectorDatabase


class FAQRetriever(BaseHybridRetriever):
    """
    Concrete implementation of a hybrid retriever for FAQ retrieval.
    
    This class implements the methods to retrieve FAQs from a vector database
    using both dense and sparse embeddings.
    
    Example:
    >>> retriever = DocumentRetriever("collection_name", embedder, bm25_client, vector_db)
    >>> results = retriever.retrieve_faqs("How to apply for admission?", top_k=3)
    """

    def __init__(
        self,
        collection_name: str,
        embedder: EmbedderCore,
        bm25_client: BM25Client,
        vector_db: VectorDatabase
    ):
        super().__init__(collection_name, embedder, bm25_client, vector_db)
        self.embedding_fields = {
            "dense": "question_dense_embedding",
            "sparse": "question_sparse_embedding"
        }
        self.output_fields = ["faq_id", "faq"]

    def retrieve_faqs(self, query: str, top_k: int = 5) -> FAQRetrievalResult:
        """
        Retrieve FAQs from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            top_k (int): The number of top FAQs to retrieve. Defaults to 5.
        
        Returns:
            FAQRetrievalResult: Result contains the query and a list of retrieved FAQs.
        """
        # Perform hybrid search
        results = self.retrieve(
            query=query,
            field_names=self.embedding_fields,
            output_fields=self.output_fields,
            top_k=top_k
        )

        retrieved_nodes = FAQRetrievalResult(
            query=query,
            faqs=[
                RetrievedFAQ(
                    source_node=FAQNode(
                        id=result["entity"]["faq_id"],
                        question=result["entity"]["faq"]["question"],
                        answer=result["entity"]["faq"]["answer"],
                    ),
                    score=result.get("distance", 0.0)
                ) for result in results
            ]
        )

        return retrieved_nodes
    
    def get_field_names(self) -> Dict[str, str]:
        """Get the field names for the embeddings used in the retrieval."""
        return self.embedding_fields
    
    def get_output_fields(self) -> List[str]:
        """Get the output fields for the retrieval results."""
        return self.output_fields
    

if __name__ == "__main__":
    import json
    from minio import Minio

    from chatbot.config.system_config import SETTINGS
    from chatbot.core.model_clients.load_model import init_embedder
    from chatbot.utils.base_class import ModelsConfig

    # Example usage
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        models_config = ModelsConfig.from_dict(models_config)

    # Initialize the embedder
    embedder = init_embedder(models_conf=models_config)

    # Initialize the MinIO client for loading BM25 state dicts
    minio_client = Minio(
        endpoint="localhost:9000",
        access_key=SETTINGS.MINIO_ACCESS_KEY_ID,
        secret_key=SETTINGS.MINIO_SECRET_ACCESS_KEY,
        secure=False
    )
    bm25_client = BM25Client(
        storage=minio_client,
        bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
        init_without_load=False,
        remove_after_load=True
    )

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host="localhost",
        port=19530,
        run_async=False
    )
    
    retriever = FAQRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
        embedder=embedder,
        bm25_client=bm25_client,
        vector_db=vector_db
    )
    
    # Example query
    query = "Học phí chương trình đại trà của trường đại học Bách Khoa 1 năm là bao nhiêu vậy?"
    results = retriever.retrieve_faqs(query, top_k=2)

    print("Retrieved results:")
    for result in results.faqs:
        print("-" * 20)
        print(f"FAQ ID: {result.source_node.id}")
        print(f"Question: {result.source_node.question}")
        print(f"Answer: {result.source_node.answer}")
        print(f"Score: {result.score}\n")
