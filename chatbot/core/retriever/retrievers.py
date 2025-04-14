from typing import Dict, List
import logging

from chatbot.core.model_clients import BM25Client, EmbedderCore
from chatbot.utils.database_clients import VectorDatabase
from chatbot.utils.database_clients.base_class import EmbeddingData

# Turn off logging for OpenAI calls
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class BaseHybridRetriever:
    """
    Base class for retrieving documents using a hybrid search approach combining dense and sparse embeddings.
    
    This class serves as a foundation for implementing various retrieval strategies using vector databases.
    It supports hybrid search combining embedding-based semantic similarity (dense) with keyword-based 
    relevance (sparse/BM25) to achieve more accurate retrieval results.
    
    Subclasses should implement specific retrieval strategies for different data types,
    such as FAQ documents, web content, or context documents, by overriding methods as needed.
    
    Attributes:
        collection_name (str): Name of the collection in the vector database.
        embedder (EmbedderCore): Model for generating dense embeddings.
        bm25_client (BM25Client): Client for generating sparse/BM25 embeddings.
        vector_db (VectorDatabase): Vector database client for performing searches.
    
    Methods:
        retrieve: Retrieve relevant documents based on a query.
        get_field_names: Get the field names for dense and sparse embeddings (to be implemented by subclasses).
        get_output_fields: Get the fields to include in the output (to be implemented by subclasses).
        process_results: Process the retrieved results (to be implemented by subclasses).
    
    Example:
        >>> embedder = EmbedderCore()
        >>> bm25_client = BM25Client()
        >>> vector_db = VectorDatabase()
        >>> retriever = ConcreteRetriever("collection_name", embedder, bm25_client, vector_db)
        >>> results = retriever.retrieve("How to apply for admission?", top_k=3)
    """
    def __init__(
        self,
        collection_name: str,
        embedder: EmbedderCore,
        bm25_client: BM25Client,
        vector_db: VectorDatabase
    ):
        self.collection_name = collection_name
        self.embedder = embedder
        self.bm25_client = bm25_client
        self.vector_db = vector_db

    def retrieve(
        self,
        query: str,
        field_names: Dict[str, str],
        output_fields: List[str],
        top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve documents from the vector database based on the query.
        
        Args:
            query (str): The query string to search for.
            field_names (Dict[str, str]): A dictionary mapping embedding types 
                (`"dense"` and `"sparse"`) to expected field names.
            output_fields (List[str]): The fields to include in the output.
            top_k (int): The number of top documents to retrieve. Defaults to 5.
        
        Returns:
            List[dict]: A list of dictionaries containing the expected output 
                fields of the retrieved items.
        """
        # Embed the query
        query_dense_embedding = self.embedder.get_query_embedding(query)
        query_sparse_embeddings = self.bm25_client.encode_queries([query])

        # Prepare the embedding data
        dense_data = EmbeddingData(
            field_name=field_names["dense"],
            embeddings=[query_dense_embedding]
        )
        sparse_data = EmbeddingData(
            field_name=field_names["sparse"],
            embeddings=query_sparse_embeddings
        )

        # Perform hybrid search
        results = self.vector_db.hybrid_search_vectors(
            collection_name=self.collection_name,
            dense_data=dense_data,
            sparse_data=sparse_data,
            output_fields=output_fields,
            top_k=top_k
        )
        
        return results[0]
        

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
        bucket_name=SETTINGS.MINIO_BUCKET_INDEX_NAME,
        init_without_load=False,
        remove_after_load=True
    )

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host="localhost",
        port=19530,
        run_async=False
    )
    
    retriever = BaseHybridRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
        embedder=embedder,
        bm25_client=bm25_client,
        vector_db=vector_db
    )
    
    # Example query

    query = "Thầy Quản Thành Thơ là ai?"
    field_names = {
        "dense": "question_dense_embedding",
        "sparse": "question_sparse_embedding"
    }
    output_fields = ["faq_id", "faq"]
    
    results = retriever.retrieve(query, field_names, output_fields, top_k=2)

    print("Retrieved results:")
    for result in results:
        print("-" * 20)
        print(f"ID: {result['faq_id']}")
        print(f"Question: {result['entity']['faq']['question']}")
        print(f"Answer: {result['entity']['faq']['answer']}")
