import os
from typing import List, Optional

import time
import traceback
from loguru import logger
from scipy.sparse._csr import csr_array
from tqdm.auto import tqdm

from minio import deleteobjects, Minio, S3Error
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction


class BM25Client:
    """
    Client for BM25-based text embedding operations.
    
    This class provides a wrapper around Milvus's BM25EmbeddingFunction with additional
    functionality for loading models from either local storage or MinIO cloud storage,
    fitting models on data, and encoding documents and queries.
    
    Attributes:
        analyzer (Analyzer): Text analyzer for tokenizing documents
        bm25 (BM25EmbeddingFunction): BM25EmbeddingFunction instance for generating embeddings
        storage_client (Minio): MinIO client for cloud storage operations
        bucket_name (str): MinIO bucket name for storing/retrieving models
    """
    def __init__(
        self,
        language: str = "en",
        local_path: Optional[str] = None,
        storage: Optional[Minio] = None,
        bucket_name: Optional[str] = None,
        remove_after_load: bool = False,
        init_without_load: bool = True,
        overwrite_minio_bucket: bool = False
    ) -> None:
        """
        Initialize the BM25Client.
        
        Args:
            language (str): Language code for the text analyzer (default: "en")
            local_path (str): Path to local BM25 state dictionary file to be loaded (optional)
            storage (Minio): MinIO client for cloud storage operations (optional)
            bucket_name (str): MinIO bucket name for storing/retrieving models (optional)
            remove_after_load (bool): Whether to delete the local copy after loading (default: False)
                               Only applies when loading from MinIO storage
            init_without_load (bool): If True, skip loading the model from local or MinIO (default: True)
            overwrite_minio_bucket (bool): If True, overwrite the existing bucket in MinIO (default: False)
        
        Raises:
            AssertionError: If neither local_path nor (storage and bucket_name) are provided
            ValueError: If BM25 state dictionary cannot be loaded
        """
        self.analyzer = build_default_analyzer(language=language)
        self.bm25 = BM25EmbeddingFunction(analyzer=self.analyzer)
        self.storage_client = storage
        self.bucket_name = bucket_name

        assert local_path or (storage and bucket_name), "Please provide the path to the BM25 state dict or the storage client and bucket name!!!"

        if init_without_load:
            if overwrite_minio_bucket and self.storage_client and self.bucket_name:
                # List all objects in the bucket with the prefix "bm25/"
                objects = list(self.storage_client.list_objects(self.bucket_name, prefix="bm25/", recursive=True))

                # Check if the state dict already exists
                if objects:
                    logger.info(f"Bucket '{self.bucket_name}' with prefix 'bm25/' already exists. Overwriting...")

                    # Clean up the folder if it exists
                    try:
                        # Remove all objects with the prefix "bm25/"
                        progress_bar = tqdm(total=len(objects), desc="Removing objects from MinIO bucket", unit="object")
                        for obj in objects:
                            self.storage_client.remove_object(self.bucket_name, obj.object_name)
                            progress_bar.update(1)
                        progress_bar.close()

                        logger.info(f"Removed existing objects in bucket '{self.bucket_name}' with prefix 'bm25/'")
                    except S3Error as e:
                        logger.warning(f"Failed to remove objects in bucket '{self.bucket_name}': {e}")
                        logger.error(traceback.format_exc())

            return # Skip loading the model if init_without_load is True

        try:
            # Load the BM25 state dict if local_path is provided
            if local_path:
                self.bm25.load(local_path)
            elif self.bucket_name and self.storage_client:
                if not os.path.exists("./bm25_state_dict.json"):
                    logger.info("Downloading the BM25 state dict...")

                    # Download the state dict from MinIO
                    retry = 3
                    while not os.path.exists("./bm25_state_dict.json"):
                        try:
                            self.storage_client.fget_object(bucket_name=self.bucket_name, object_name="bm25/state_dict.json", file_path="./bm25_state_dict.json")
                        except Exception as e:
                            logger.warning(f"Failed to download the BM25 state dict: {e}")

                            # Wailt for the file to be ready if exists
                            for i in range(5):
                                if os.path.exists("./bm25_state_dict.json"):
                                    break
                                time.sleep(1)

                            # Retry downloading if the file is not found
                            if not os.path.exists("./bm25_state_dict.json"):
                                logger.warning("Failed to download the BM25 state dict. Retrying...")
                                retry -= 1

                                # If retries are exhausted, raise an error
                                if retry == 0:
                                    raise ValueError("Failed to download the BM25 state dict")

                logger.info("Loading the BM25 state dict...")
                if not os.path.exists("./bm25_state_dict.json"):
                    raise ValueError("BM25 state dict not found in the local directory")
                self.bm25.load("./bm25_state_dict.json")

                # Remove the state dict if exists
                if os.path.exists("./bm25_state_dict.json") and remove_after_load:
                    logger.info("Removing the BM25 state dict in the local directory...")
                    os.remove("./bm25_state_dict.json")
            else:
                raise ValueError("Please provide the path to the BM25 state dict or the storage client and bucket name")
        except Exception as e:
            logger.error(f"Failed to load the BM25 state dict: {e}")
            logger.error(traceback.format_exc())

    def _load_from_local(self, local_path: str) -> None:
        """
        Load BM25 model from a local file path.
        
        Args:
            local_path (str): Path to the local BM25 state dictionary file
            
        Raises:
            ValueError: If the file cannot be loaded
        """
        logger.info(f"Loading BM25 state dict from {local_path}")
        self.bm25.load(local_path)
        
    def _load_from_minio(self, remove_after_load: bool) -> None:
        """
        Download and load BM25 model from MinIO storage.
        
        Args:
            remove_after_load (bool): Whether to delete the local copy after loading
            
        Raises:
            ValueError: If download fails after retries or file cannot be loaded
        """
        local_path = "./bm25_state_dict.json"
        
        # Download if local copy doesn't exist
        if not os.path.exists(local_path):
            logger.info("Downloading BM25 state dict from MinIO...")
            self._download_from_minio(local_path)
                
        # Load the dictionary
        logger.info("Loading the BM25 state dict...")
        if not os.path.exists(local_path):
            raise ValueError("BM25 state dict not found in the local directory")
        
        self.bm25.load(local_path)

        # Remove the local copy if requested
        if remove_after_load and os.path.exists(local_path):
            logger.info("Removing the local BM25 state dict...")
            os.remove(local_path)

    def _download_from_minio(self, local_path: str, max_retries: int = 3) -> None:
        """
        Download BM25 state dictionary from MinIO with retries.
        
        Args:
            local_path (str): Path to save the downloaded file
            max_retries (int): Maximum number of download attempts
            
        Raises:
            ValueError: If download fails after all retries
        """
        for retry in range(max_retries):
            try:
                self.storage_client.fget_object(
                    bucket_name=self.bucket_name, 
                    object_name="bm25/state_dict.json", 
                    file_path=local_path
                )
                
                # Check if download was successful
                if os.path.exists(local_path):
                    return
                    
            except Exception as e:
                logger.warning(f"Download attempt {retry+1} failed: {e}")
                
                # Check if file appeared after exception
                for i in range(5):
                    if os.path.exists(local_path):
                        return
                    time.sleep(1)
            
            logger.warning(f"Retry {retry+1}/{max_retries} for downloading BM25 state dict")
            
        raise ValueError(f"Failed to download BM25 state dict after {max_retries} attempts")

    def fit(self, data: List[str], path: str = "./bm25_state_dict.json", auto_save_local: bool = False) -> None:
        """
        Train the BM25 model on the provided text data and save the state dictionary.
        
        Args:
            data (List[str]): List of text documents to train on
            path (str): Path to save the BM25 state dictionary
            auto_save_local (bool): Whether to automatically save the model locally
            
        Raises:
            ValueError: If storage client and bucket name are not available for uploading
        """
        # Train the model
        logger.info(f"Fitting BM25 model on {len(data)} documents")
        self.bm25.fit(data)
        
        # Ensure directory exists and save locally
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.bm25.save(path)

        # Upload to MinIO if storage is configured
        if not (self.storage_client and self.bucket_name):
            logger.warning("Storage client or bucket name not provided - state dict saved locally only")
            return
            
        logger.info(f"Uploading BM25 state dict to MinIO bucket '{self.bucket_name}'")
        self.storage_client.fput_object(
            bucket_name=self.bucket_name,
            object_name="bm25/state_dict.json",
            file_path=path
        )

        if auto_save_local:
            logger.info(f"Saved BM25 state dict to {path}")
        else:
            # Remove the local copy if auto_save_local is False
            if os.path.exists(path):
                os.remove(path)
            logger.info(f"BM25 state dict saved to MinIO bucket '{self.bucket_name}'")

    def fit_transform(self, data: List[str], path: str = "./bm25_state_dict.json", auto_save_local: bool = False) -> List[csr_array]:
        """
        Train the BM25 model on the provided text data and transform the documents to embeddings.
        
        Args:
            data (List[str]): List of text documents to train on and transform
            path (str): Path to save the BM25 state dictionary
            auto_save_local (bool): Whether to automatically save the model locally
            
        Returns:
            List of BM25 embeddings for the input documents
        """
        # Fit the model and save state
        self.fit(data, path, auto_save_local)
        
        # Transform the documents
        logger.info(f"Encoding {len(data)} documents with BM25")
        return self.encode_text(data)
    
    def encode_text(self, data: List[str]) -> List[csr_array]:
        """
        Convert text documents to BM25 embeddings.
        
        Args:
            data (List[str]): List of text documents to encode
            
        Returns:
            List[csr_array]: List of BM25 embeddings for the input documents
        """
        return list(self.bm25.encode_documents(data))

    def encode_queries(self, queries: List[str]) -> List[csr_array]:
        """
        Convert search queries to BM25 embeddings.
        
        Args:
            queries (List[str]): List of search queries to encode
            
        Returns:
            List[csr_array]: List of BM25 embeddings for the input queries
        """
        return list(self.bm25.encode_queries(queries))
    