from typing import Dict, List

from chatbot.core.retriever.base_class import (
    DocumentNode,
    RetrievedDocument,
    DocumentRetrievalResult
)
from chatbot.core.retriever.base_retriever import BaseHybridRetriever
from chatbot.core.model_clients import BM25Client, EmbedderCore
from chatbot.utils.database_clients import VectorDatabase


class DocumentRetriever(BaseHybridRetriever):
    """
    Concrete implementation of a hybrid retriever for document retrieval.
    
    This class implements the methods to retrieve documents from a vector database
    using both dense and sparse embeddings.
    
    Example:
    >>> retriever = DocumentRetriever("collection_name", embedder, bm25_client, vector_db)
    >>> results = retriever.retrieve_documents("How to apply for admission?", top_k=3)
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
            "dense": "chunk_dense_embedding",
            "sparse": "chunk_sparse_embedding"
        }
        self.output_fields = ["chunk_id", "chunk"]

    def retrieve_documents(self, query: str, top_k: int = 5) -> DocumentRetrievalResult:
        """
        Retrieve documents from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            top_k (int): The number of top documents to retrieve. Defaults to 5.
        
        Returns:
            DocumentRetrievalResult: Result contains the query and a list of retrieved documents.
        """
        # Perform hybrid search
        results = self.retrieve(
            query=query,
            field_names=self.embedding_fields,
            output_fields=self.output_fields,
            top_k=top_k
        )

        retrieved_nodes = DocumentRetrievalResult(
            query=query,
            documents=[
                RetrievedDocument(
                    source_node=DocumentNode(
                        id=result["entity"]["chunk_id"],
                        chunk=result["entity"]["chunk"]
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
        bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
        init_without_load=False,
        remove_after_load=True
    )

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host="localhost",
        port=19530,
        run_async=False
    )
    
    retriever = DocumentRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
        embedder=embedder,
        bm25_client=bm25_client,
        vector_db=vector_db
    )
    
    # Example query
    query = "Trường đại học Bách Khoa có bao nhiêu ngành giảng dạy bằng tiếng anh?"
    results = retriever.retrieve_documents(query, top_k=2)

    print("Retrieved results:")
    for result in results.documents:
        print("-" * 20)
        print(f"Document ID: {result.source_node.id}")
        print(f"Document Content: {result.source_node.chunk}")
        print(f"Score: {result.score}\n")
