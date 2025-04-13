from chatbot.config.system_config import SETTINGS
from chatbot.utils.base_class import ModelsConfig
from chatbot.core.chat_stores import CacheChatStore
from chatbot.core.model_clients import EmbedderCore, LLMCore


def init_llm(
    task: str,
    models_conf: ModelsConfig,
    cache_chat_store: CacheChatStore = None,
    user_id: str = None,
    assistant_id : str = None
) -> LLMCore:
    """
    Initialize the LLM model for a specific task.
    
    Args:
        task (str): The task for which to initialize the LLM.
        models_conf (ModelsConfig): The configuration for the models.
        cache_chat_store (CacheChatStore, optional): The cache chat store. Defaults to None.
        user_id (str, optional): The user ID. Defaults to None.
        assistant_id (str, optional): The assistant ID. Defaults to None.

    Returns:
        LLMCore: The initialized LLM model.
    """
    if task not in models_conf.llm_config:
        raise ValueError(f"Task '{task}' is not defined in the model configuration.")

    config = models_conf.llm_config[task]
    use_openai = (config.provider == "openai")

    API_KEY = SETTINGS.OTHER_API_KEY if not use_openai else SETTINGS.OPENAI_API_KEY

    return LLMCore(
        user_id=user_id,
        assistant_id =assistant_id,
        model_id=config.model_id,
        chat_store=cache_chat_store,
        OPENAI_API_KEY=API_KEY,
        use_openai=use_openai,
        base_url=config.base_url,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        is_chat=config.is_chat,
        use_websocket=config.use_websocket
    )


def init_embedder(models_conf: ModelsConfig) -> EmbedderCore:
    """
    Initialize the Embedder model.
    
    Args:
        models_conf (ModelsConfig): The configuration for the models.
    Returns:
        EmbedderCore: The initialized Embedder model.
    """
    if not models_conf.embedding_config:
        raise ValueError("Embedding model configuration is missing.")

    config = models_conf.embedding_config
    use_openai = (config.provider == "openai")

    return EmbedderCore(
        uri=config.base_url,
        model_id=config.model_id,
        OPENAI_API_KEY=SETTINGS.OPENAI_API_KEY if use_openai else "EMPTY",
        use_openai=use_openai
    )
