from dataclasses import dataclass


@dataclass
class FAQDocument:
    """
    FAQ document containing question and answer pairs.

    Args:
        id (str): Unique identifier for the FAQ document (UUID).
        question (str): The question in the FAQ.
        answer (str): The answer to the question.
    """
    id: str
    question: str
    answer: str
