import json
import uuid
from typing import Dict, List
from tqdm.auto import tqdm

from chatbot.core.model_clients import LLMCore
from chatbot.indexing.context_document.base_class import ReconstructedChunk
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.prompts.indexing.generate_faq import FAQ_GENERATION_PROMPT_TEMPLATE


class FaqGenerator:
    """
    A class that generates FAQ pairs from reconstructed text chunks using a large language model.
    
    This class processes text chunks by:
    1. Analyzing the content of each chunk to identify potential questions users might ask
    2. Generating appropriate and accurate answers based on the chunk's information
    3. Creating structured FAQ pairs that capture the most relevant information
    
    The generation process transforms informational content into a question-answer format
    that is more accessible and directly addresses potential user queries.
    
    Attributes:
        llm (LLMCore): The language model used for FAQ generation.
    
    Methods:
        generate_faq: Process reconstructed chunks to create FAQ pairs.
    
    Example:
        >>> llm = LLMCore()
        >>> chunks = [ReconstructedChunk(document="doc1", chunk="Information about admissions")]
        >>> generator = FaqGenerator(llm)
        >>> faq_documents = generator.generate_faq(chunks, max_pairs=5)
    """
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def generate_faq(
        self,
        documents: List[ReconstructedChunk],
        max_pairs: int = 5
    ) -> List[FAQDocument]:
        """
        Generate a document from the FAQ data.

        Args:
            documents (List[ReconstructedChunk]): List of reconstructed chunks.
            max_pairs (int): Maximum number of FAQ pairs to generate.

        Returns:
            List[FAQDocument]: List of generated FAQ documents.
        """
        total_pairs: List[Dict[str, str]] = []
        progress_bar = tqdm(documents, desc="Generating FAQ pairs")
        # Iterate through each reconstructed chunk
        for document in documents:
            response = self.llm.complete(
            prompt=FAQ_GENERATION_PROMPT_TEMPLATE.format(
                    text_chunk=document.chunk,
                    max_faq_pairs=max_pairs
                )
            ).text

            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")

            # Parse the JSON response and append to the total pairs
            total_pairs.extend(json.loads(response))

            # Update progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Convert the list of pairs into a list of FAQDocument objects
        list_faq = [
            FAQDocument(
                id=str(uuid.uuid4()),
                question=pair["question"],
                answer=pair["answer"]
            )
            for pair in total_pairs
        ]

        return list_faq
