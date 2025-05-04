from typing import List
from tqdm.auto import tqdm

import logging
import pandas as pd
from llama_index.core import Document
from loguru import logger

from chatbot.core.model_clients import BM25Client, EmbedderCore, LLMCore
from chatbot.indexing.context_document.base_class import PreprocessingConfig, ReconstructedChunk
from chatbot.indexing.context_document import (
    ChunkReconstructor,
    ContextExtractor,
    SemanticChunker
)
from chatbot.indexing.faq import (
    FaqAugmenter,
    FaqExpander,
    FaqGenerator
)
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.utils.base_class import IndexData
from chatbot.utils.database_clients.base_class import IndexParam, IndexValueType
from chatbot.utils.database_clients import VectorDatabase


# Turn off logging for OpenAI calls
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class DataIndex:
    def __init__(
        self,
        llm: LLMCore,
        embedder: EmbedderCore,
        document_bm25_client: BM25Client,
        faq_bm25_client: BM25Client,
        preprocessing_config: PreprocessingConfig,
        vector_db: VectorDatabase
    ):
        """
        Initialize the DataIndex class.
        This class is responsible for indexing documents and FAQ context into a vector database.

        Args:
            llm (LLMCore): Language model used for context extraction.
            embedder (EmbedderCore): Embedder model to embed the documents into dense vectors.
            document_bm25_client (BM25Client): BM25 client to embed the documents into sparse vectors.
            faq_bm25_client (BM25Client): BM25 client to embed the FAQs into sparse vectors.
            preprocessing_config (PreprocessingConfig): Configuration for preprocessing text.
            vector_db (VectorDatabase): Vector database client for storing indexed data.
        """
        self.semantic_chunker = SemanticChunker(
            embedder=embedder,
            preprocessing_config=preprocessing_config
        )
        self.context_extractor = ContextExtractor(llm)
        self.chunk_reconstructor = ChunkReconstructor(llm)
        self.faq_expander = FaqExpander(llm)
        self.faq_generator = FaqGenerator(llm)
        self.faq_augmenter = FaqAugmenter(llm)
        self.vector_db = vector_db
        self.embedder = embedder
        self.document_bm25_client = document_bm25_client
        self.faq_bm25_client = faq_bm25_client

    def run_index(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> IndexData:
        """
        Run the indexing process for documents and FAQs.

        Args:
            documents (List[str]): List of documents to index.
            faqs (List[FAQDocument]): List of FAQ documents to index.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """
        # Build index
        index_data = self.build_index(
            documents=documents,
            faqs=faqs,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name,
            overwrite_collection=True
        )

        # Index data
        self.index(
            index_data=index_data,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name
        )

        return index_data
    
    def run_insert(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> IndexData:
        """
        Run the inserting process for documents and FAQs.

        Args:
            documents (List[str]): List of documents to insert.
            faqs (List[FAQDocument]): List of FAQ documents to insert.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """
        # Build index for new data
        index_data = self.build_index(
            documents=documents,
            faqs=faqs,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name,
            overwrite_collection=False
        )

        # Insert the new index data into the vector database
        self.insert(
            index_data=index_data,
            document_collection_name=document_collection_name,
            faq_collection_name=faq_collection_name
        )

        return index_data

    def index(
        self,
        index_data: IndexData,
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> None:
        """
        Index the given data into the vector database.

        Args:
            index_data (IndexData): Data to be indexed.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """
        # Index documents
        if index_data.documents:
            logger.info(f"Indexing {len(index_data.documents)} documents...")
            progress_bar = tqdm(index_data.documents, desc="Indexing documents")

            # Fit the BM25 client to the documents
            chunk_sparse_embeddings = self.document_bm25_client.fit_transform(
                [document.chunk for document in index_data.documents]
            )

            for idx, document in enumerate(index_data.documents):
                # Generate dense embeddings for the chunks
                dense_embedding = self.embedder.get_text_embedding(document.chunk)

                # Index the document into the vector database
                self.vector_db.insert_vectors(
                    collection_name=document_collection_name,
                    data={
                        "chunk_id": document.id,
                        "chunk": document.chunk,
                        "chunk_dense_embedding": dense_embedding,
                        "chunk_sparse_embedding": chunk_sparse_embeddings[idx]
                    }
                )

                # Update progress bar
                progress_bar.update(1)

            # Close the progress bar
            progress_bar.close()

        # Index FAQs
        if index_data.faqs:
            logger.info(f"Indexing {len(index_data.faqs)} FAQs...")
            progress_bar = tqdm(index_data.faqs, desc="Indexing FAQs")

            # Fit the BM25 client to the FAQs
            faq_sparse_embeddings = self.faq_bm25_client.fit_transform(
                [(faq.question + " " + faq.answer) for faq in index_data.faqs]
            )

            for idx, faq in enumerate(index_data.faqs):
                # Generate dense embeddings for the FAQ
                combined_faq = faq.question + " " + faq.answer
                dense_embedding = self.embedder.get_text_embedding(combined_faq)

                # Index the FAQ into the vector database
                self.vector_db.insert_vectors(
                    collection_name=faq_collection_name,
                    data={
                        "faq_id": faq.id,
                        "faq": {
                            "question": faq.question,
                            "answer": faq.answer
                        },
                        "question_dense_embedding": dense_embedding,
                        "question_sparse_embedding": faq_sparse_embeddings[idx]
                    }
                )

                # Update progress bar
                progress_bar.update(1)

            # Close the progress bar
            progress_bar.close()

        logger.info("Indexing completed.")

    def insert(
        self,
        index_data: IndexData,
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection"
    ) -> None:
        """
        Insert the new index data into the vector database, updating the old index data.

        Args:
            index_data (IndexData): New data to be inserted.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
        """

        logger.info("Inserting new index data and updating old index data...")

        # Get indexed dense embeddings
        chunk_data = self.vector_db.get_all_vectors(
            collection_name=document_collection_name,
            field_names=["chunk_id", "chunk", "chunk_dense_embedding"]
        )
        faq_data = self.vector_db.get_all_vectors(
            collection_name=faq_collection_name,
            field_names=["faq_id", "faq", "question_dense_embedding"]
        )
        
        # Store the dense embeddings in a dictionary
        chunk_embedding_dict = {
            chunk["chunk_id"]: chunk["chunk_dense_embedding"] for chunk in chunk_data
        }
        faq_embedding_dict = {
            faq["faq_id"]: faq["question_dense_embedding"] for faq in faq_data
        }

        # Add dense embeddings for the new data
        dense_embeddings = self.embedder.get_text_embedding_batch(
            [document.chunk for document in index_data.documents]
        )
        for idx, document in enumerate(index_data.documents):
            # Update the dense embedding in the dictionary
            chunk_embedding_dict[document.id] = dense_embeddings[idx]

        dense_embeddings = self.embedder.get_text_embedding_batch(
            [(faq.question + " " + faq.answer) for faq in index_data.faqs]
        )
        for idx, faq in enumerate(index_data.faqs):
            # Update the dense embedding in the dictionary
            faq_embedding_dict[faq.id] = dense_embeddings[idx]

        # Update BM25 state dicts
        total_chunks = [document.chunk for document in index_data.documents] + [data["chunk"] for data in chunk_data]
        chunk_sparse_embeddings = self.document_bm25_client.fit_transform(
            data=total_chunks
        )
        total_faqs = [(faq.question + " " + faq.answer) for faq in index_data.faqs] + [data["faq"]["question"] for data in faq_data]
        faq_sparse_embeddings = self.faq_bm25_client.fit_transform(
            data=total_faqs
        )

        # Add old chunk data into index data
        index_data.documents.extend([
            ReconstructedChunk(
                id=data["chunk_id"],
                chunk= data["chunk"]
            )
            for data in chunk_data
        ])
        # Add old FAQ data into index data
        index_data.faqs.extend([
            FAQDocument(
                id=data["faq_id"],
                question=data["faq"]["question"],
                answer=data["faq"]["answer"]
            )
            for data in faq_data
        ])

        # Clear the old data in the vector database
        logger.info("Deleting old data in the vector database...")
        self._create_collection(document_collection_name, faq_collection_name)

        # Index all chunk data into the vector database
        for idx, document in enumerate(index_data.documents):
            # Get dense embedding from the dictionary
            dense_embedding = chunk_embedding_dict[document.id]

            # Index the document into the vector database
            self.vector_db.insert_vectors(
                collection_name=document_collection_name,
                data={
                    "chunk_id": document.id,
                    "chunk": document.chunk,
                    "chunk_dense_embedding": dense_embedding,
                    "chunk_sparse_embedding": chunk_sparse_embeddings[idx]
                }
            )

        # Index all FAQ data into the vector database
        for idx, faq in enumerate(index_data.faqs):
            # Get dense embedding from the dictionary
            dense_embedding = faq_embedding_dict[faq.id]

            # Index all FAQs into the vector database
            self.vector_db.insert_vectors(
                collection_name=faq_collection_name,
                data={
                    "faq_id": faq.id,
                    "faq": {
                        "question": faq.question,
                        "answer": faq.answer
                    },
                    "question_dense_embedding": dense_embedding,
                    "question_sparse_embedding": faq_sparse_embeddings[idx]
                }
            )

    def build_index(
        self,
        documents: List[str] = [],
        faqs: List[FAQDocument] = [],
        document_collection_name: str = "document_collection",
        faq_collection_name: str = "faq_collection",
        overwrite_collection: bool = True
    ) -> IndexData:
        """
        Build index for documents and FAQ context.

        Args:
            documents (List[str]): List of documents to index.
            faqs (List[FAQDocument]): List of FAQ documents to index.
            document_collection_name (str): Name of the collection for documents.
            faq_collection_name (str): Name of the collection for FAQs.
            overwrite_collection (bool): Flag to overwrite existing collections.

        Returns:
            IndexData: Data to be indexed containing reconstructed chunks and FAQs.
        """
        # ------ Create collections ------
        if overwrite_collection:
            self._create_collection(document_collection_name, faq_collection_name)

        # ------ Build document context ------
        logger.info(f"Generating document context for {len(documents)} documents...")

        # Initialize empty list for reconstructed chunks
        reconstructed_chunks = []

        progress_bar = tqdm(documents, desc="Building document context")
        for document in documents:
            # Step 1: Extract context from documents
            extracted_context = self.context_extractor.extract_context_documents([document])

            # Step 2: Chunk documents
            text_chunks = self.semantic_chunker.chunk([Document(text=document)])

            # Step 3: Reconstruct documents
            reconstructed_chunks.extend(self.chunk_reconstructor.reconstruct_chunks(
                chunks=text_chunks,
                context=extracted_context[0]
            ))

            # Update progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # ------ Build FAQ context ------
        logger.info(f"Generating FAQ context for {len(faqs)} FAQs and {len(reconstructed_chunks)} reconstructed chunks...")
        
        # Initialize empty list for FAQs if not provided
        processed_faqs = faqs

        # Step 1: Expand FAQ pairs from existing ones
        if processed_faqs:
            expanded_faqs = self.faq_expander.expand_faq(processed_faqs)
            processed_faqs.extend(expanded_faqs)


        # Step 2: Generate new FAQ pairs from reconstructed chunks
        if reconstructed_chunks:
            generated_faqs = self.faq_generator.generate_faq(reconstructed_chunks)
            processed_faqs.extend(generated_faqs)

        # Step 3: Enrich FAQ pairs
        if processed_faqs:
            enriched_faqs = self.faq_augmenter.augment_faq(processed_faqs)
            processed_faqs.extend(enriched_faqs)

        return IndexData(
            documents=reconstructed_chunks,
            faqs=processed_faqs
        )

    def _create_collection(self, document_collection_name: str, faq_collection_name: str):
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): Name of the collection to create.
        """
        logger.info(f"Creating collection {document_collection_name} in the vector database...")

        # Define the chunk collection schema
        chunk_collection_structure = [
            {
                "field_name": "chunk_id",
                "field_type": "string",
                "field_description": "ID of the chunk",
                "is_primary": True
            },
            {
                "field_name": "chunk",
                "field_type": "string",
                "field_description": "Content of the chunk"
            },
            {
                "field_name": "chunk_dense_embedding",
                "field_type": "float",
                "field_description": "Dense embedding of the chunk",
                "dimension": len(self.embedder.get_text_embedding("test")),
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "index_params": {"M": 16, "efConstruction": 500}
            },
            {
                "field_name": "chunk_sparse_embedding",
                "field_type": "sparse_float",
                "field_description": "Sparse embedding of the chunk",
            }
        ]

        # Create the document collection in the vector database
        self.vector_db.create_collection(
            collection_name=document_collection_name,
            collection_structure=chunk_collection_structure
        )

        logger.info(f"Creating collection {faq_collection_name} in the vector database...")

        # Define the FAQ collection schema
        faq_collection_structure = [
            {
                "field_name": "faq_id",
                "field_type": "string",
                "field_description": "ID of the FAQ",
                "is_primary": True
            },
            {
                "field_name": "faq",
                "field_type": "json",
                "field_description": "Content of the FAQ",
            },
            {
                "field_name": "question_dense_embedding",
                "field_type": "float",
                "field_description": "Dense embedding of the question",
                "dimension": len(self.embedder.get_text_embedding("test")),
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "index_params": {"M": 16, "efConstruction": 500}
            },
            {
                "field_name": "question_sparse_embedding",
                "field_type": "sparse_float",
                "field_description": "Sparse embedding of the question",
            }
        ]

        # Define the JSON index parameters for the FAQ collection
        json_index_params = {
            "faq": [
                IndexParam(
                    indexed_key="faq['question']",
                    index_name="question",
                    value_type=IndexValueType.STRING
                ),
                IndexParam(
                    indexed_key="faq['answer']",
                    index_name="answer",
                    value_type=IndexValueType.STRING
                ),
            ]
        }

        # Create the FAQ collection in the vector database
        self.vector_db.create_collection(
            collection_name=faq_collection_name,
            collection_structure=faq_collection_structure,
            json_index_params=json_index_params
        )
        

if __name__ == "__main__":
    import json
    import uuid
    from minio import Minio
    from chatbot.config.system_config import SETTINGS
    from chatbot.core.model_clients.load_model import init_embedder, init_llm
    from chatbot.utils.base_class import ModelsConfig

    # Example usage

    # Load model configurations
    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        models_config = ModelsConfig.from_dict(models_config)

    # Initialize the MinIO client for storing BM25 state dicts
    minio_client = Minio(
        endpoint="localhost:9000",
        access_key=SETTINGS.MINIO_ACCESS_KEY_ID,
        secret_key=SETTINGS.MINIO_SECRET_ACCESS_KEY,
        secure=False
    )

    # Initialize the LLM and embedder clients
    llm = init_llm(
        task="indexing_llm",
        models_conf=models_config
    )
    embedder = init_embedder(models_conf=models_config)
    document_bm25_client = BM25Client(
        storage=minio_client,
        bucket_name=SETTINGS.MINIO_BUCKET_DOCUMENT_INDEX_NAME,
        overwrite_minio_bucket=True
    )
    faq_bm25_client = BM25Client(
        storage=minio_client,
        bucket_name=SETTINGS.MINIO_BUCKET_FAQ_INDEX_NAME,
        overwrite_minio_bucket=True
    )

    # Initialize the configuration for preprocessing before chunking
    preprocessing_config = PreprocessingConfig()

    # Initialize the vector database client
    vector_db = VectorDatabase(
        host="localhost",
        port=19530,
        run_async=False
    )

    # Initialize the Indexer
    indexer = DataIndex(
        llm=llm,
        embedder=embedder,
        document_bm25_client=document_bm25_client,
        faq_bm25_client=faq_bm25_client,
        preprocessing_config=preprocessing_config,
        vector_db=vector_db
    )
    
    # Load documents from file text
    open_file_path = "./chatbot/.data/test_document.txt"
    with open(open_file_path, "r") as file:
        document = file.read()
    documents = [document]

    # Load FAQs from file csv
    open_faq_path = "./chatbot/.data/test_faq.csv"
    faq_df = pd.read_csv(open_faq_path)
    FAQs = [
        FAQDocument(
            id=str(uuid.uuid4()),
            question=row["query"],
            answer=row["answer"]
        )
        for _, row in faq_df.iterrows()
    ]

    result = indexer.run_index(
        documents=documents,
        faqs=FAQs,
        document_collection_name=SETTINGS.MILVUS_COLLECTION_DOCUMENT_NAME,
        faq_collection_name=SETTINGS.MILVUS_COLLECTION_FAQ_NAME
    )

    # Display the result
    print("\nReconstructed Chunks:")
    for chunk in result.documents:
        print("--" * 20)
        print(f"ID: {chunk.id}")
        print(f"Document: {chunk.document}")
        print(f"Chunk: {chunk.chunk}")

    print("\nGenerated FAQs:")
    for faq in result.faqs:
        print("--" * 20)
        print(f"ID: {faq.id}")
        print(f"Question: {faq.question}")
        print(f"Answer: {faq.answer}")
