import json
from tqdm import tqdm
from typing import List

from chatbot.core.model_clients import LLMCore
from chatbot.indexing.context_document.base_class import ExtractedContext
from chatbot.prompts.indexing.extract_context import CONTEXT_EXTRACTION_PROMPT_TEMPLATE


class ContextExtractor:
    def __init__(self, llm: LLMCore):
        self.llm = llm

    def extract_context_documents(self, documents: List[str]) -> List[ExtractedContext]:
        """
        Extract context from a list of documents.

        Args:
            documents: List of documents to extract context from.

        Returns:
            contexts (List[ExtractedContext]): List of contexts extracted from the documents.
        """
        contexts: List[ExtractedContext] = []
        progress_bar = tqdm(documents, desc="Extracting context from documents")
        for document in documents:
            # Extract context from the document
            context = self.extract_context_single_document(document)

            contexts.append(context)
            
            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        return contexts

    def extract_context_single_document(self, document: str) -> ExtractedContext:
        """
        Extract context from a single document.
        
        Args:
            document (str): Document to extract context from.
            
        Returns:
            context (ExtractedContext): Context extracted from the document.
        """
        response = self.llm.complete(
            prompt=CONTEXT_EXTRACTION_PROMPT_TEMPLATE.format(
                text=document,
                max_tokens=self.llm.max_new_tokens
            )
        ).text

        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "")
        
        summary_context = json.loads(response)["summary"]

        return ExtractedContext(
            document=document,
            context=summary_context
        )
        