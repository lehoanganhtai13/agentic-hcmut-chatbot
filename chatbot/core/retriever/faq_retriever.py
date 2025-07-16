from typing import Dict, List, Optional

from chatbot.core.retriever.base_class import (
    FAQNode,
    RetrievedFAQ,
    FAQRetrievalResult
)
from chatbot.core.retriever.base_retriever import BaseHybridRetriever
from chatbot.core.model_clients import BM25Client, BaseEmbedder, BaseReranker
from chatbot.utils.database_clients import BaseVectorDatabase


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
        embedder: BaseEmbedder,
        bm25_client: BM25Client,
        vector_db: BaseVectorDatabase,
        reranker: Optional[BaseReranker] = None
    ):
        super().__init__(collection_name, embedder, bm25_client, vector_db)
        self.embedding_fields = {
            "dense": "question_dense_embedding",
            "sparse": "question_sparse_embedding"
        }
        self.output_fields = ["faq_id", "faq"]
        self.reranker = reranker

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
            top_k=(max(20, top_k) if self.reranker else top_k)
        )

        # If reranker is configured, rerank the results
        if self.reranker:
            reranked_results = self.reranker.rerank(
                query=query,
                documents=[
                    (result["faq"]["question"] + ": " + result["faq"]["answer"])
                    for result in results
                ]
            )
            # Sort results based on reranked scores
            results = sorted(
                zip(results, reranked_results),
                key=lambda x: x[1][1],  # Sort by relevance score
                reverse=True
            )
            results = [result[0] for result in results[:top_k]]

        retrieved_nodes = FAQRetrievalResult(
            query=query,
            faqs=[
                RetrievedFAQ(
                    source_node=FAQNode(
                        id=result["faq_id"],
                        question=result["faq"]["question"],
                        answer=result["faq"]["answer"],
                    ),
                    score=result.get("_score", 0.0)
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

    from chatbot.config.system_config import SETTINGS
    from chatbot.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
    from chatbot.core.model_clients.reranker.vllm import VLLMRerankerConfig, VLLMReranker
    from chatbot.utils.base_class import ModelsConfig
    from chatbot.utils.database_clients.milvus import MilvusVectorDatabase, MilvusConfig

    # Example usage
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        embedder_config = ModelsConfig.from_dict(models_config).embedding_config
        reranker_config = ModelsConfig.from_dict(models_config).reranker_config

    if embedder_config.provider != "openai":
        raise ValueError("Supported provider is OpenAI only for this example.")

    # Initialize the embedder
    embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        use_openai_client=(models_config.embedding_config.provider == "openai"),
        base_url= embedder_config.base_url,
        query_embedding_endpoint="v1/embeddings",
        doc_embedding_endpoint="v1/embeddings"
    ))

    # Initialize the reranker
    reranker = VLLMReranker(
        config=VLLMRerankerConfig(
            base_url=reranker_config.base_url,
            rerank_endpoint=reranker_config.rerank_endpoint
    ))

    # Initialize BM25 client
    bm25_client = BM25Client(
        local_path="./chatbot/data/bm25/faq/state_dict.json",
        init_without_load=False
    )

    # Initialize the vector database client
    vector_db = MilvusVectorDatabase(
        config=MilvusConfig(
            cloud_uri=SETTINGS.MILVUS_CLOUD_URI,
            token=SETTINGS.MILVUS_CLOUD_TOKEN,
            run_async=False
        )
    )
    
    retriever = FAQRetriever(
        collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME,
        embedder=embedder,
        bm25_client=bm25_client,
        vector_db=vector_db,
        reranker=reranker
    )
    
    # Example query
    query = "Học phí chương trình tiêu chuẩn của trường đại học Bách Khoa 1 năm là bao nhiêu vậy?"
    results = retriever.retrieve_faqs(query, top_k=1)

    print("Retrieved results:")
    for result in results.faqs:
        print("-" * 20)
        print(f"FAQ ID: {result.source_node.id}")
        print(f"Question: {result.source_node.question}")
        print(f"Answer: {result.source_node.answer}")
        print(f"Score: {result.score}\n")
