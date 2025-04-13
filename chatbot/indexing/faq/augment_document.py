import json
import uuid
from typing import List
from tqdm.auto import tqdm

from chatbot.core.model_clients import LLMCore
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.prompts.indexing.paraphrase_faq import FAQ_PARAPHRASE_PROMPT_TEMPLATE


class FaqAugmenter:
    """
    A class that augments FAQ documents by generating additional FAQs based on existing ones.
    
    This class processes FAQ documents to create new FAQ pairs by paraphrasing or rephrasing
    existing questions and answers, enhancing the diversity and coverage of the FAQ content.
    
    Attributes:
        llm (LLMCore): The language model used for FAQ augmentation.
    
    Methods:
        augment_faq: Process existing FAQ documents to create augmented FAQ pairs.
    
    Example:
        >>> llm = LLMCore()
        >>> faq_documents = [FAQDocument(question="What is AI?", answer="AI is artificial intelligence.")]
        >>> augmenter = FaqAugmenter(llm)
        >>> augmented_faqs = augmenter.augment_faq(faq_documents, max_pairs=3)
    """
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def augment_faq(
        self,
        documents: List[FAQDocument],
        max_pairs: int = 3
    ) -> List[FAQDocument]:
        """
        Augment FAQ documents by generating additional FAQ pairs.
        
        Args:
            documents (List[FAQDocument]): List of existing FAQ documents.
            max_pairs (int): Maximum number of FAQ pairs to generate for each document.
        
        Returns:
            List[FAQDocument]: List of augmented FAQ documents.
        """
        augmented_faqs: List[FAQDocument] = []
        progress_bar = tqdm(documents, desc="Augmenting FAQ pairs")
        
        # Iterate through each FAQ document
        for document in documents:
            # Generate new Questions based on the existing the original FAQ pair
            response = self.llm.complete(
                prompt=FAQ_PARAPHRASE_PROMPT_TEMPLATE.format(
                    faq_pair={"question": document.question, "answer": document.answer},
                    max_paraphrases=max_pairs
                )
            ).text
            
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            
            # Parse the JSON response
            list_new_question = json.loads(response)
            
            # Add new FAQ pairs to the list
            for new_question in list_new_question:
                augmented_faqs.append(FAQDocument(
                    id=str(uuid.uuid4()),
                    question=new_question["paraphrased_question"],
                    answer=document.answer
                ))
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()
        
        return augmented_faqs
