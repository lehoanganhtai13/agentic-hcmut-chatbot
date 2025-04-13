import json
import uuid
from typing import List
from tqdm.auto import tqdm

from chatbot.core.model_clients import LLMCore
from chatbot.indexing.context_document.base_class import ExtractedContext, ReconstructedChunk
from chatbot.prompts.indexing.generate_title import GENERATE_TITLE_QUICK_DESCRIPTION_PROMPT_TEMPLATE
from chatbot.prompts.indexing.rewrite_chunk import REWRITE_TEXT_CHUNK_PROMPT_TEMPLATE


class ChunkReconstructor:
    """
    A class that reconstructs text chunks using a large language model for enhanced readability and context.
    
    This class processes text chunks by:
    1. Rewriting them with additional contextual information
    2. Generating relevant titles or descriptions for each chunk
    3. Combining titles with rewritten content to create contextualized chunks
    
    The reconstruction process ensures that individual text chunks maintain relevance 
    to the original document context while being independently comprehensible.
    
    Attributes:
        llm (LLMCore): The language model used for text rewriting and title generation.
    
    Methods:
        reconstruct_chunks: Process multiple chunks with context to create enhanced text chunks.
        rewrite_chunk: Transform an individual chunk using document context.
        generate_title_quick_description: Create a concise title for a chunk.
        combine_title_and_chunk: Format title and content into a single string.
    
    Example:
        >>> llm = LLMCore()
        >>> context = ExtractedContext(document="original doc", context="summary context")
        >>> reconstructor = ChunkReconstructor(llm)
        >>> chunks = ["text chunk 1", "text chunk 2"]
        >>> reconstructed = reconstructor.reconstruct_chunks(chunks, context)
    """
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def reconstruct_chunks(self, chunks: List[str], context: ExtractedContext) -> List[ReconstructedChunk]:
        """
        Reconstruct a list of chunks using the LLM model.

        Args:
            chunks (List[str]): Chunks to reconstruct.
            context (ExtractedContext): Context used to reconstruct the chunks.

        Returns:
            reconstructed_chunks (List[ReconstructedChunk]): Reconstructed chunks.
        """
        progress_bar = tqdm(chunks, desc="Reconstructing chunks")
        reconstructed_chunks = []
        for chunk in chunks:
            # Step 1: Rewrite the chunk
            rewritten_chunk = self.rewrite_chunk(chunk, context)

            # Step 2: Generate a title or quick description for the rewritten chunk
            title = self.generate_title_quick_description(rewritten_chunk)

            # Step 3: Combine the title and chunk
            title_and_chunk = self.combine_title_and_chunk(title, rewritten_chunk)

            reconstructed_chunks.append(ReconstructedChunk(
                id=str(uuid.uuid4()),
                document=context.document,
                chunk=title_and_chunk
            ))

            # Update progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        
        return reconstructed_chunks

    def rewrite_chunk(self, chunk: str, context: ExtractedContext) -> str:
        """
        Rewrite a chunk using the LLM model.

        Args:
            chunk (str): Chunk to rewrite.
            context (ExtractedContext): Context usd to rewrite the chunk.

        Returns:
            rewritten_chunk (str): Rewritten chunk.
        """
        response = self.llm.complete(
            prompt=REWRITE_TEXT_CHUNK_PROMPT_TEMPLATE.format(
                text_chunk=chunk,
                context=context.context,
                max_tokens=self.llm.max_new_tokens
            )
        ).text

        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")

        rewritten_chunk = json.loads(response)["rewritten_chunk"]
        return rewritten_chunk
    
    def generate_title_quick_description(self, rewritten_chunk: str) -> str:
        """
        Generate a title or quick description for a rewritten chunk.

        Args:
            rewritten_chunk (str): Rewritten chunk to generate title or quick description for.

        Returns:
            title_or_quick_description (str): Title or quick description.
        """
        response = self.llm.complete(
            prompt=GENERATE_TITLE_QUICK_DESCRIPTION_PROMPT_TEMPLATE.format(
                rewritten_chunk=rewritten_chunk
            )
        ).text

        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")

        title_or_quick_description = json.loads(response)["title_or_quick_description"]

        # Remove trailing period
        if title_or_quick_description.endswith("."):
            title_or_quick_description = title_or_quick_description[:-1]

        return title_or_quick_description
    
    def combine_title_and_chunk(self, title: str, chunk: str) -> str:
        """
        Combine a title with a chunk.

        Args:
            title (str): Title.
            chunk (str): Chunk.

        Returns:
            title_and_chunk (str): Title and chunk combined.
        """
        return f"# {title}\n\n{chunk}"
