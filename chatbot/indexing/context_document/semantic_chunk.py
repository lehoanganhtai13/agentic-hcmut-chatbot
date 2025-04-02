from typing import List
import re

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser

from chatbot.core.model_clients import EmbedderCore
from chatbot.indexing.context_document.base_class import PreprocessingConfig


class SemanticChunker:
    def __init__(
        self,
        embedder: EmbedderCore,
        preprocessing_config: PreprocessingConfig,
        similarity_percentile_threshold: int = 90
    ):
        """
        Initialize the SemanticChunker. This class chunks documents into smaller parts based on semantic similarity.

        Attributes:
            embedder (EmbedderCore): Embedder model to embed the documents.
            preprocessing_config (PreprocessingConfig): Configuration for preprocessing text.
            similarity_percentile_threshold (int): Percentile threshold for semantic similarity.
        """
        self.embedder = embedder
        self.preprocessing_config = preprocessing_config
        self.chunker = SemanticSplitterNodeParser(
            embed_model=embedder,
            breakpoint_percentile_threshold=similarity_percentile_threshold
        )

    def chunk(self, documents: List[Document]) -> List[str]:
        """
        Chunk a list of documents into smaller parts based on semantic similarity.

        Args:
            documents (List[Document]): List of documents to chunk.

        Returns:
            chunks (List[str]): List of chunks.
        """
        # Preprocess documents
        preprocessed_documents = self.preprocess(documents)

        # Chunk documents
        nodes = self.chunker.get_nodes_from_documents(preprocessed_documents)

        return [node.get_content().strip() for node in nodes]
    
    async def chunk_async(self, documents: List[Document]) -> List[str]:
        """
        Chunk a list of documents into smaller parts based on semantic similarity asynchronously.

        Args:
            documents (List[Document]): List of documents to chunk.

        Returns:
            chunks (List[str]): List of chunks.
        """
        # Preprocess documents
        preprocessed_documents = self.preprocess(documents)

        # Chunk documents asynchronously
        nodes = await self.chunker.aget_nodes_from_documents(preprocessed_documents)

        return [node.get_content().strip() for node in nodes]

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess a list of documents.

        Args:
            documents (List[Document]): List of documents to preprocess.

        Returns:
            documents (List[Document]): Preprocessed list of documents.
        """
        for doc in documents:
            raw_text = doc.get_content()
            
            # Clean whitespace
            if self.preprocessing_config.clean_whitespace:
                raw_text = self._clean_whitespace(raw_text)

            # Clean empty lines
            if self.preprocessing_config.clean_empty_lines:
                raw_text = self._clean_empty_lines(raw_text)

            # Clean header and footer
            if self.preprocessing_config.clean_header_footer:
                raw_text = self._clean_header_footer(raw_text)

            # Remove URLs
            if self.preprocessing_config.remove_urls:
                raw_text = self._remove_urls(raw_text)

            # Remove HTML tags
            if self.preprocessing_config.remove_html_tags:
                raw_text = self._remove_html_tags(raw_text)

            # Normalize unicode
            if self.preprocessing_config.normalize_unicode:
                raw_text = self._normalize_unicode(raw_text)

            # Remove custom patterns
            if self.preprocessing_config.custom_patterns:
                raw_text = self._remove_custom_patterns(raw_text, self.preprocessing_config.custom_patterns)

            doc.set_content(raw_text)

        return documents

    def _clean_whitespace(self, text: str) -> str:
        """Clean whitespace in a text."""
        pages = text.split("\f")
        for page in pages:
            lines = page.splitlines()

            # Replace multiple spaces with single space and strip leading/trailing spaces
            cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]

        return "\f".join(cleaned_lines)
    
    def _clean_empty_lines(self, text: str) -> str:
        """Clean empty lines in a text."""
        return re.sub(r"\n\n\n+", "\n\n", text)
    
    def _clean_header_footer(self, text: str) -> str:
        """Clean header and footer in a text by removing lines that contain only numbers or hyphens."""
        text = re.sub(r"^[\d-]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Page \d+$", "", text, flags=re.MULTILINE)
        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs in a text."""
        return re.sub(r"https?://\S+|www\.\S+", "", text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags in a text."""
        return re.sub(r"<.*?>", "", text)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode in a text."""
        import unicodedata
        return unicodedata.normalize("NFC", text)
    
    def _remove_custom_patterns(self, text: str, patterns: List[str]) -> str:
        """Remove custom patterns in a text."""
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text
    