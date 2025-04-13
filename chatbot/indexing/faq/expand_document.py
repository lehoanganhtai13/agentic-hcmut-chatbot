import json
import uuid
from typing import List
from tqdm.auto import tqdm

from chatbot.core.model_clients import LLMCore
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.prompts.indexing.expand_faq import FAQ_DETAIL_EXPANSION_PROMPT_TEMPLATE


class FaqExpander:
    """
    A class that expands FAQ documents by generating additional FAQs based on existing ones.
    
    This class processes FAQ documents to create new FAQ pairs that are related to the original content.
    
    Attributes:
        llm (LLMCore): The language model used for FAQ expansion.
    
    Methods:
        expand_faq: Process existing FAQ documents to create expanded FAQ pairs.
    
    Example:
        >>> llm = LLMCore()
        >>> faq_documents = [FAQDocument(question="What is AI?", answer="AI is artificial intelligence.")]
        >>> expander = FaqExpander(llm)
        >>> expanded_faqs = expander.expand_faq(faq_documents, max_pairs=3)
    """
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def expand_faq(
        self,
        documents: List[FAQDocument],
        max_pairs: int = 3
    ) -> List[FAQDocument]:
        """
        Expand FAQ documents by generating additional FAQ pairs.
        
        Args:
            documents (List[FAQDocument]): List of existing FAQ documents.
            max_pairs (int): Maximum number of FAQ pairs to generate for each document.
        
        Returns:
            List[FAQDocument]: List of expanded FAQ documents.
        """
        expanded_faqs: List[FAQDocument] = []
        progress_bar = tqdm(documents, desc="Expanding FAQ pairs")
        
        # Iterate through each FAQ document
        for document in documents:
            response = self.llm.complete(
                prompt=FAQ_DETAIL_EXPANSION_PROMPT_TEMPLATE.format(
                    faq_pair={"question": document.question, "answer": document.answer},
                    max_new_faq_pairs=max_pairs
                )
            ).text
            
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            
            # Parse the response to extract FAQ pairs
            faq_pairs = json.loads(response)
            
            for pair in faq_pairs:
                expanded_faqs.append(FAQDocument(
                    id=str(uuid.uuid4()),
                    question=pair["question"],
                    answer=pair["answer"]
                ))
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()
        
        return expanded_faqs
