from loguru import logger
from tqdm import tqdm
from typing import List

from chatbot.core.model_clients import EmbedderCore, LLMCore
from chatbot.indexing.context_document.base_class import PreprocessingConfig, ReconstructedChunk
from chatbot.indexing.context_document.extract_context import ContextExtractor
from chatbot.indexing.context_document.reconstruct_chunk import ChunkReconstructor
from chatbot.indexing.context_document.semantic_chunk import SemanticChunker


class ContextDocumentIndexer:
    def __init__(
        self,
        llm: LLMCore,
        embedder: EmbedderCore,
        preprocessing_config: PreprocessingConfig,
        similarity_percentile_threshold: int = 90
    ):
        """
        Initialize the ContextDocumentIndexer. This class indexes documents into smaller parts based on semantic similarity.

        Attributes:
            llm (LLMCore): Language model used for context extraction.
            embedder (EmbedderCore): Embedder model to embed the documents.
            preprocessing_config (PreprocessingConfig): Configuration for preprocessing text.
            similarity_percentile_threshold (int): Percentile threshold for semantic similarity.
        """
        self.llm = llm
        self.embedder = embedder
        self.preprocessing_config = preprocessing_config
        self.semantic_chunker = SemanticChunker(
            embedder=embedder,
            preprocessing_config=preprocessing_config,
            similarity_percentile_threshold=similarity_percentile_threshold
        )
        self.context_extractor = ContextExtractor(llm)
        self.chunk_reconstructor = ChunkReconstructor(llm)

    def index_documents(self, documents: List[str]) -> List[ReconstructedChunk]:
        """
        Index documents into smaller parts based on semantic similarity.

        Args:
            documents (List[str]): List of documents to index.

        Returns:
            indexed_documents (List[ReconstructedChunk]): List of indexed documents.
        """
        logger.info(f"Starting indexing of {len(documents)} documents...")

        # Step 1: Extract context from documents
        extracted_contexts = self.context_extractor.extract_context_documents(documents)

        # Reconstruct documents
        indexed_documents: List[ReconstructedChunk] = []
        progress_bar = tqdm(extracted_contexts, desc="Indexing documents with context")
        for context in extracted_contexts:
            # Step 2: Chunk documents
            text_chunks = self.semantic_chunker.chunk(context.document)

            # Step 3: Reconstruct chunks
            reconstructed_chunks = self.chunk_reconstructor.reconstruct_chunks(text_chunks, context)

            # Add reconstructed chunks to indexed documents
            indexed_documents.extend(reconstructed_chunks)

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        return indexed_documents