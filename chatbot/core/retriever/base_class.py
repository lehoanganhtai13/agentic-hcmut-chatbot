from typing import List
from dataclasses import dataclass

from chatbot.indexing.context_document.base_class import ReconstructedChunk
from chatbot.indexing.faq.base_class import FAQDocument


DocumentNode = ReconstructedChunk
FAQNode = FAQDocument


@dataclass
class RetrievedDocument:
    """
    A class to represent a retrieved document.

    Args:
        source_node (DocumentNode): The source document node.
        score (float): Relevance score of the document.
    """

    source_node: DocumentNode
    score: float


@dataclass
class DocumentRetrievalResult:
    """
    A class to represent the result of a document retrieval operation.

    Args:
        query (str): The query used for retrieval.
        documents (List[RetrievedDocument]): A list of retrieved documents.
    """

    query: str
    documents: List[RetrievedDocument]


@dataclass
class RetrievedFAQ:
    """
    A class to represent the result of a FAQ retrieval operation.

    Args:
        source_node (FAQNode): The source FAQ node.
        score (float): Relevance score of the FAQ.
    """

    source_node: FAQNode
    score: float


@dataclass
class FAQRetrievalResult:
    """
    A class to represent the result of a FAQ retrieval operation.

    Args:
        query (str): The query used for retrieval.
        faqs (List[RetrievedFAQ]): A list of retrieved FAQs.
    """

    query: str
    faqs: List[RetrievedFAQ]
