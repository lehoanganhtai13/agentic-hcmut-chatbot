from typing import Callable, List, Optional

import re
import numpy as np
from dataclasses import dataclass
from loguru import logger
from tqdm.auto import tqdm

from chatbot.core.model_clients import BaseEmbedder
from chatbot.indexing.context_document.base_class import PreprocessingConfig
from chatbot.utils.embeddings import DenseEmbedding


@dataclass
class SentenceGroup:
    """
    Data class representing a group of sentences with their embeddings.
    
    Attributes:
        sentence (str): The original sentence.
        index (int): Index of the sentence in the document.
        combined_sentence (str): Combined sentence with buffer context.
        embedding (Optional[DenseEmbedding]): Embedding vector for the combined sentence.
    """
    sentence: str
    index: int
    combined_sentence: str
    embedding: Optional[DenseEmbedding] = None


class SemanticChunker:
    def __init__(
        self,
        embedder: BaseEmbedder,
        preprocessing_config: PreprocessingConfig,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        min_chunk_size: int = 10,
        max_chunk_size: int = 1000
    ):
        """
        Initialize the SemanticChunker. This class chunks documents into smaller parts based on semantic similarity.

        This chunker uses sentence-level embeddings and cosine similarity to identify semantic
        boundaries in text, creating chunks that maintain topical coherence rather than
        arbitrary length-based splits.
        
        The algorithm works by:
        1. Splitting text into sentences
        2. Creating sentence groups with buffer context
        3. Computing embeddings for each group
        4. Calculating semantic distances between consecutive groups
        5. Identifying breakpoints based on percentile thresholds
        6. Forming chunks at identified breakpoints

        Attributes:
            embedder (BaseEmbedder): Embedder model to embed the documents.
            preprocessing_config (PreprocessingConfig): Configuration for preprocessing text.
            buffer_size (int): Number of sentences to include in the buffer context when computing embeddings.
            breakpoint_percentile_threshold (int): Percentile threshold for identifying semantic breakpoints.
                Lower values create more chunks.
            sentence_splitter (Optional[Callable[[str], List[str]]]): Function to split text into sentences.
            min_chunk_size (int): Minimum size of a chunk in characters.
            max_chunk_size (int): Maximum size of a chunk in characters.
        """
        if buffer_size < 0:
            raise ValueError("Buffer size must be non-negative")

        if not 0 <= breakpoint_percentile_threshold <= 100:
            raise ValueError("Breakpoint percentile threshold must be between 0 and 100")

        self.embedder = embedder
        self.preprocessing_config = preprocessing_config
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Initialize sentence splitter
        self.sentence_splitter = sentence_splitter or self._default_sentence_splitter

        logger.info(
            f"🔧 Initialized SemanticChunker with buffer_size={buffer_size}, "
            f"threshold={breakpoint_percentile_threshold}%, "
            f"chunk_size_range=({min_chunk_size}, {max_chunk_size})"
        )

    def chunk(self, documents: List[str]) -> List[str]:
        """
        Chunk a list of documents into smaller parts based on semantic similarity.

        Args:
            documents (List[str]): List of documents to chunk.

        Returns:
            chunks (List[str]): List of chunks.
        """
        # Preprocess documents
        preprocessed_documents = self.preprocess(documents)

        # Chunk documents
        all_chunks = []
        progress_bar = tqdm(total=len(preprocessed_documents), desc="Chunking documents")
        for doc in preprocessed_documents:
            doc_chunks = self._chunk_single_document(doc)
            all_chunks.extend(doc_chunks)

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        # Filter chunks by size constraints
        filtered_chunks = self._filter_chunks_by_size(all_chunks)

        return filtered_chunks

    async def chunk_async(self, documents: List[str]) -> List[str]:
        """
        Chunk a list of documents into smaller parts based on semantic similarity asynchronously.

        Args:
            documents (List[str]): List of documents to chunk.

        Returns:
            chunks (List[str]): List of chunks.
        """
        # Preprocess documents
        preprocessed_documents = self.preprocess(documents)

        # Chunk documents asynchronously
        all_chunks = []
        for doc in preprocessed_documents:
            doc_chunks = await self._chunk_single_document_async(doc)
            all_chunks.extend(doc_chunks)

        # Filter chunks by size constraints
        filtered_chunks = self._filter_chunks_by_size(all_chunks)

        return filtered_chunks

    def _chunk_single_document(self, text: str) -> List[str]:
        """
        Chunk a single document using semantic similarity analysis.
        
        Args:
            text (str): Text content to chunk.
            
        Returns:
            List[str]: List of semantic chunks from the document.
        """
        # Split text into sentences
        sentences = self.sentence_splitter(text)
        
        if not sentences:
            logger.warning("No sentences found in document")
            return []
        
        if len(sentences) == 1:
            logger.debug("Document contains only one sentence")
            return [text.strip()]
        
        # Build sentence groups with buffer context
        sentence_groups = self._build_sentence_groups(sentences)
        
        # Get embeddings for sentence groups
        combined_sentences = [group.combined_sentence for group in sentence_groups]
        embeddings = self.embedder.get_text_embeddings(combined_sentences)
        
        # Assign embeddings to sentence groups
        for group, embedding in zip(sentence_groups, embeddings):
            group.embedding = embedding
        
        # Calculate semantic distances
        distances = self._calculate_semantic_distances(sentence_groups)
        
        # Build chunks based on semantic breakpoints
        chunks = self._build_chunks_from_breakpoints(sentence_groups, distances)
        
        return chunks
    
    async def _chunk_single_document_async(self, text: str) -> List[str]:
        """
        Asynchronously chunk a single document using semantic similarity analysis.
        
        Args:
            text (str): Text content to chunk.
            
        Returns:
            List[str]: List of semantic chunks from the document.
        """
        # Split text into sentences
        sentences = self.sentence_splitter(text)
        
        if not sentences:
            logger.warning("No sentences found in document")
            return []
        
        if len(sentences) == 1:
            logger.debug("Document contains only one sentence")
            return [text.strip()]
        
        # Build sentence groups with buffer context
        sentence_groups = self._build_sentence_groups(sentences)
        
        # Get embeddings for sentence groups asynchronously
        combined_sentences = [group.combined_sentence for group in sentence_groups]
        embeddings = await self.embedder.aget_text_embeddings(combined_sentences)
        
        # Assign embeddings to sentence groups
        for group, embedding in zip(sentence_groups, embeddings):
            group.embedding = embedding
        
        # Calculate semantic distances
        distances = self._calculate_semantic_distances(sentence_groups)
        
        # Build chunks based on semantic breakpoints
        chunks = self._build_chunks_from_breakpoints(sentence_groups, distances)
        
        return chunks

    def _build_sentence_groups(self, sentences: List[str]) -> List[SentenceGroup]:
        """
        Build sentence groups with buffer context for more accurate semantic analysis.
        
        The buffer provides context around each sentence to improve embedding quality
        and semantic boundary detection.
        
        Args:
            sentences (List[str]): List of sentences to group.
            
        Returns:
            List[SentenceGroup]: List of sentence groups with buffer context.
        """
        sentence_groups = []
        
        for i, sentence in enumerate(sentences):
            # Calculate buffer range
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(len(sentences), i + self.buffer_size + 1)
            
            # Combine sentences within buffer range
            buffer_sentences = sentences[start_idx:end_idx]
            combined_sentence = " ".join(buffer_sentences).strip()
            
            group = SentenceGroup(
                sentence=sentence,
                index=i,
                combined_sentence=combined_sentence
            )
            sentence_groups.append(group)

        return sentence_groups

    def _calculate_semantic_distances(self, sentence_groups: List[SentenceGroup]) -> List[float]:
        """
        Calculate semantic distances between consecutive sentence groups using cosine similarity.
        
        Args:
            sentence_groups (List[SentenceGroup]): List of sentence groups with embeddings.
            
        Returns:
            List[float]: List of semantic distances (1 - cosine_similarity).
        """
        distances = []
        
        for i in range(len(sentence_groups) - 1):
            current_embedding = sentence_groups[i].embedding
            next_embedding = sentence_groups[i + 1].embedding
            
            if current_embedding is None or next_embedding is None:
                logger.warning(f"Missing embedding for sentence group at index {i}")
                distances.append(0.0)
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(current_embedding, next_embedding)
            
            # Convert to distance (higher distance = less similar)
            distance = 1.0 - similarity
            distances.append(distance)
        
        logger.debug(f"Calculated {len(distances)} semantic distances")
        return distances

    def _cosine_similarity(self, vec1: DenseEmbedding, vec2: DenseEmbedding) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1 (DenseEmbedding): First vector.
            vec2 (DenseEmbedding): Second vector.
            
        Returns:
            float: Cosine similarity value between -1 and 1.
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        magnitude1 = np.linalg.norm(vec1_np)
        magnitude2 = np.linalg.norm(vec2_np)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return float(similarity)

    def _build_chunks_from_breakpoints(
        self, 
        sentence_groups: List[SentenceGroup], 
        distances: List[float]
    ) -> List[str]:
        """
        Build text chunks based on semantic breakpoints identified from distance analysis.
        
        Args:
            sentence_groups (List[SentenceGroup]): List of sentence groups.
            distances (List[float]): List of semantic distances between consecutive groups.
            
        Returns:
            List[str]: List of text chunks split at semantic boundaries.
        """
        if not distances:
            # If no distances (single sentence or very short document), return entire text
            combined_text = " ".join(group.sentence for group in sentence_groups)
            return [combined_text.strip()] if combined_text.strip() else []
        
        # Calculate breakpoint threshold using percentile
        breakpoint_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
        
        # Identify indices where distance exceeds threshold
        breakpoint_indices = [
            i for i, distance in enumerate(distances) 
            if distance > breakpoint_threshold
        ]
        
        # Build chunks
        chunks = []
        start_idx = 0
        
        for breakpoint_idx in breakpoint_indices:
            # Create chunk from start_idx to breakpoint_idx (inclusive)
            chunk_groups = sentence_groups[start_idx:breakpoint_idx + 1]
            chunk_text = " ".join(group.sentence for group in chunk_groups)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            start_idx = breakpoint_idx + 1
        
        # Add remaining sentences as final chunk
        if start_idx < len(sentence_groups):
            remaining_groups = sentence_groups[start_idx:]
            chunk_text = " ".join(group.sentence for group in remaining_groups)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        # Handle very unbalanced chunks
        chunks = self._balance_chunk_sizes(chunks)
        
        return chunks
    
    def _balance_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """
        Balance chunk sizes to avoid very small final chunks.
        
        Args:
            chunks (List[str]): List of chunks to balance.
            
        Returns:
            List[str]: Balanced chunks.
        """
        if len(chunks) < 2:
            return chunks
        
        balanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # If this is the last chunk and it's significantly smaller than min_chunk_size * 2
            if (i == len(chunks) - 1 and 
                len(chunk) < self.min_chunk_size * 2 and 
                len(balanced_chunks) > 0):
                
                # Merge with previous chunk if it doesn't exceed max size
                prev_chunk = balanced_chunks[-1]
                merged_text = prev_chunk + " " + chunk
                
                if len(merged_text) <= self.max_chunk_size:
                    balanced_chunks[-1] = merged_text
                    logger.debug(f"Balanced chunks: merged small final chunk (size {len(chunk)}) with previous")
                else:
                    balanced_chunks.append(chunk)
            else:
                balanced_chunks.append(chunk)
        
        return balanced_chunks

    def _filter_chunks_by_size(self, chunks: List[str]) -> List[str]:
        """
        Filter chunks based on size constraints and merge/split as needed.
        
        Args:
            chunks (List[str]): List of chunks to filter.
            
        Returns:
            List[str]: List of size-filtered chunks.
        """
        if not chunks:
            return []
        
        # Step 1: Handle large chunks first (split them)
        size_adjusted_chunks = []
        for chunk in chunks:
            chunk_len = len(chunk)
            
            # Split chunks that are too large
            if chunk_len > self.max_chunk_size:
                logger.debug(f"Splitting large chunk of size {chunk_len}")
                sub_chunks = self._split_large_chunk(chunk)
                size_adjusted_chunks.extend(sub_chunks)
            else:
                size_adjusted_chunks.append(chunk)
        
        # Step 2: Merge small chunks with adjacent chunks
        merged_chunks = self._merge_small_chunks(size_adjusted_chunks)
        
        # Step 3: Final filtering (only discard if absolutely necessary)
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk) >= self.min_chunk_size:
                final_chunks.append(chunk)
            else:
                logger.debug(f"Discarding chunk of size {len(chunk)} (below minimum after merge attempts)")

        logger.debug(f"Filtered chunks: {len(chunks)} -> {len(final_chunks)}")
        return final_chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge small chunks with adjacent chunks to prevent data loss.
        
        Args:
            chunks (List[str]): List of chunks to process.
            
        Returns:
            List[str]: List of chunks after merging small ones.
        """
        if not chunks:
            return []
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If current chunk is large enough, keep it
            if len(current_chunk) >= self.min_chunk_size:
                merged_chunks.append(current_chunk)
                i += 1
                continue
            
            # Current chunk is too small, try to merge
            merged = False
            
            # Try to merge with next chunk
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                merged_text = current_chunk + " " + next_chunk
                
                # Check if merged chunk doesn't exceed max size
                if len(merged_text) <= self.max_chunk_size:
                    merged_chunks.append(merged_text)
                    i += 2  # Skip next chunk as it's merged
                    merged = True
                    logger.debug(f"Merged chunk {i-1} with {i} (sizes: {len(current_chunk)} + {len(next_chunk)} = {len(merged_text)})")
            
            # If couldn't merge with next, try to merge with previous
            if not merged and merged_chunks:
                prev_chunk = merged_chunks[-1]
                merged_text = prev_chunk + " " + current_chunk
                
                # Check if merged chunk doesn't exceed max size
                if len(merged_text) <= self.max_chunk_size:
                    merged_chunks[-1] = merged_text  # Replace previous chunk
                    merged = True
                    logger.debug(f"Merged chunk {i} with previous (sizes: {len(prev_chunk)} + {len(current_chunk)} = {len(merged_text)})")
            
            # If still couldn't merge, keep the small chunk (will be filtered later)
            if not merged:
                merged_chunks.append(current_chunk)
            
            i += 1
        
        return merged_chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """
        Split a large chunk into smaller pieces while preserving sentence boundaries.
        
        Args:
            chunk (str): Large chunk to split.
            
        Returns:
            List[str]: List of smaller chunks.
        """
        sentences = self.sentence_splitter(chunk)
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                sub_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add final chunk if not empty
        if current_chunk.strip():
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks

    def _default_sentence_splitter(self, text: str) -> List[str]:
        """
        Default sentence splitter using regex patterns.
        
        Args:
            text (str): Text to split into sentences.
            
        Returns:
            List[str]: List of sentences.
        """
        # Split on sentence-ending punctuation followed by whitespace or end of string
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def preprocess(self, documents: List[str]) -> List[str]:
        """
        Preprocess a list of documents.

        Args:
            documents (List[str]): List of documents to preprocess.

        Returns:
            documents (List[str]): Preprocessed list of documents.
        """
        processed_documents = []
        for doc in documents:
            raw_text = doc

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

            processed_documents.append(raw_text)

        return processed_documents

    def _clean_whitespace(self, text: str) -> str:
        """Clean whitespace in a text."""
        pages = text.split("\f")
        cleaned_pages = []
        
        for page in pages:
            lines = page.splitlines()
            # Replace multiple spaces with single space and strip leading/trailing spaces
            cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
            cleaned_pages.append("\n".join(cleaned_lines))
        
        return "\f".join(cleaned_pages)
    
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
    

if __name__ == "__main__":
    # Example usage
    import json

    from chatbot.config.system_config import SETTINGS
    from chatbot.core.model_clients.embedder.openai import OpenAIClientConfig, OpenAIEmbedder
    from chatbot.utils.base_class import ModelsConfig

    models_config = {}
    with open("./chatbot/config/models_config.json", "r") as f:
        # Load the JSON file
        models_config = json.load(f)

        # Convert the loaded JSON to a ModelsConfig object
        embedder_config = ModelsConfig.from_dict(models_config).embedding_config

    if embedder_config.provider != "openai":
        raise ValueError("Supported provider is OpenAI only for this example.")

    # Initialize the embedder
    embedder = OpenAIEmbedder(config=OpenAIClientConfig(
        api_key=SETTINGS.OPENAI_API_KEY,
        model=embedder_config.model_id,
    ))

    preprocessing_config = PreprocessingConfig()
    
    chunker = SemanticChunker(
        embedder, 
        preprocessing_config,
        buffer_size=1,
        breakpoint_percentile_threshold=80,  # Lower threshold for better chunking
        min_chunk_size=100,
        max_chunk_size=1000
    )
    documents = [
        """
        Artificial Intelligence has revolutionized many industries in recent years. From healthcare to finance, AI technologies are being deployed to solve complex problems and improve efficiency. Machine learning algorithms can now process vast amounts of data and identify patterns that would be impossible for humans to detect.
        
        However, the rapid advancement of AI also brings significant challenges. Privacy concerns have become more prominent as AI systems collect and analyze personal data. There are also questions about job displacement as automation replaces human workers in various sectors.
        
        The ethical implications of AI development cannot be ignored. Bias in algorithmic decision-making has led to discriminatory outcomes in hiring, lending, and criminal justice systems. Researchers and policymakers are working to develop frameworks for responsible AI development.
        
        Looking ahead, the future of AI depends on how well we can balance innovation with ethical considerations. International cooperation will be essential to establish global standards for AI safety and security. Education and retraining programs will help workers adapt to the changing job market.
        
        Ultimately, the goal should be to harness AI's potential while minimizing its risks. This requires ongoing dialogue between technologists, ethicists, policymakers, and the public. Only through collaborative effort can we ensure that AI serves humanity's best interests.
        """
    ]
    chunks = chunker.chunk(documents)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")
    