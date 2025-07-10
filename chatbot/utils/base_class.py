from pydantic import BaseModel
from typing import Dict, List, Optional

from chatbot.indexing.context_document.base_class import ReconstructedChunk
from chatbot.indexing.faq.base_class import FAQDocument


class IndexData(BaseModel):
    """
    A class representing the data structure for indexing documents.

    Attributes:
        documents (List[ReconstructedChunk]): List of reconstructed chunks to be indexed.
        faqs (List[FAQDocument]): List of FAQ documents to be indexed.
    """
    documents: Optional[List[ReconstructedChunk]] = None
    faqs: Optional[List[FAQDocument]] = None

    class Config:
        arbitrary_types_allowed = True


class LLMConfig(BaseModel):
    """
    A class representing the configuration for a model.

    Attributes:
        model_id (str): The ID of the model.
        provider (str): The provider of the model.
        base_url (str): The base URL for the model.
        max_new_tokens (int): The maximum number of new tokens.
        temperature (float): The temperature for the model.
        thinking_mode (bool): Whether the model is in thinking mode (default is False).
    """
    model_id: str
    provider: str
    base_url: str
    max_new_tokens: int
    temperature: float
    thinking_mode: bool = False


class EmbeddingConfig(BaseModel):
    """
    A class representing the configuration for an embedding model.

    Attributes:
        model_id (str): The ID of the embedding model.
        provider (str): The provider of the embedding model.
        base_url (Optional[str]): The base URL for the embedding model (not required for OpenAI).
    """
    model_id: str
    provider: str
    base_url: Optional[str] = None


class ModelsConfig(BaseModel):
    """
    A class representing the configuration for models.

    Attributes:
        llm_config (dict): A dictionary containing configurations for different LLM tasks.
        embedding (EmbeddingConfig): The configuration for the embedding model.
    """
    llm_config: Dict[str, LLMConfig]
    embedding_config: EmbeddingConfig

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Dict]) -> "ModelsConfig":
        """
        Create a ModelsConfig instance from a dictionary.

        Args:
            config_dict (Dict[str, Dict]): The configuration dictionary.

        Returns:
            ModelsConfig: An instance of ModelsConfig.
        """
        llm_config = {key: LLMConfig(**value) for key, value in config_dict["LLM"].items()}
        embedding_config = EmbeddingConfig(**config_dict["Embedding"])
        return cls(llm_config=llm_config, embedding_config=embedding_config)
