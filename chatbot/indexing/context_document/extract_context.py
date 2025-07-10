import json
from loguru import logger
from tqdm import tqdm
from typing import List

from chatbot.core.model_clients import BaseLLM
from chatbot.core.model_clients.llm.google import GoogleAIClientLLMConfig
from chatbot.indexing.context_document.base_class import ExtractedContext
from chatbot.prompts.indexing.extract_context import CONTEXT_EXTRACTION_PROMPT_TEMPLATE


class ContextExtractor:
    def __init__(self, llm: BaseLLM):
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
            try:
                # Extract context from the document
                context = self.extract_context_single_document(document)

                # Append the extracted context to the list
                if context:
                    contexts.append(context)
            except Exception as e:
                logger.error(f"Error extracting context from document: {e}")
                continue

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
        try:
            response = self.llm.complete(
                prompt=CONTEXT_EXTRACTION_PROMPT_TEMPLATE.format(
                    text=document,
                    max_tokens=self.llm.config.max_tokens,
                )
            ).text

            # Clean the response to remove code block formatting
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            
            # Parse the JSON response
            try:
                summary_context = json.loads(response)["summary"]
            except json.JSONDecodeError as e:
                logger.debug(f"Received invalid JSON response: {response}")
                logger.error(f"Error decoding JSON response: {e}")
                return None

            return ExtractedContext(
                document=document,
                context=summary_context
            )
        except Exception as e:
            logger.error(f"Error extracting context from document: {e}")
            return None