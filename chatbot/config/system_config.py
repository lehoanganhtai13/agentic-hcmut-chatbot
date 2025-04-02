import os
from dotenv import load_dotenv
from chatbot.config.utils import check_bool


class Settings:
    """Settings for the chatbot service."""
    def __init__(self):

        # Load environment variables
        env_file = os.getenv("ENVIRONMENT_FILE", "/environments/.env")
        status = load_dotenv(env_file)
        if not status:
            raise Exception(f"Could not load environment variables from {env_file}")
        
        # MinIO settings
        self.MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
        self.MINIO_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY_ID", "minioadmin")
        self.MINIO_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY", "minioadmin")

        # Milvus settings
        self.MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
        self.MILVUS_MINIO_BUCKET_NAME = os.getenv("MILVUS_MINIO_BUCKET_NAME", "milvus-data")

        # Model serving settings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "EMPTY")
        self.OTHER_API_KEY = os.getenv("OTHER_API_KEY", "EMPTY")

        # Agent settings
        self.SCRAPELESS_API_KEY = os.getenv("SCRAPELESS_API_KEY", "")

SETTINGS = Settings()
