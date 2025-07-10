import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import time
import traceback
from loguru import logger
import numpy as np
from scipy.sparse import csr_array, csr_matrix, vstack
from tqdm.auto import tqdm

from minio import Minio, S3Error
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction

from chatbot.utils.embeddings import SparseEmbedding


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
    
    Examples:

        1. Training Mode (First Time Setup):
            ```python
            # Initialize for training without pre-existing model
            client = BM25Client(
                language="en",
                storage=minio_client,
                bucket_name="ml-models",
                init_without_load=True  # Skip loading, prepare for training
            )
            
            # Train on your corpus and save to cloud
            client.fit(training_documents, auto_save_local=False)
            ```
        
        2. Inference Mode (Load Existing Model):
            ```python
            # Load pre-trained model from cloud storage
            client = BM25Client(
                language="en", 
                storage=minio_client,
                bucket_name="ml-models",
                init_without_load=False,    # Load existing model
                remove_after_load=True      # Clean up local cache
            )
            
            # Use for encoding queries/documents
            embeddings = client.encode_text(["query text"])
            ```
        
        3. Local Development Mode:
            ```python
            # Train and save locally
            client = BM25Client(
                language="en",
                init_without_load=True
            )
            client.fit(
                documents,
                path="./models/bm25_model.json",
                auto_save_local=True
            )
            ```
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
            local_path (str): Path to local BM25 state dictionary file to be loaded (example: "./bm25/state_dict.json") 
            storage (Minio): MinIO client for cloud storage operations (optional)
            bucket_name (str): MinIO bucket name for storing/retrieving models (optional)
            remove_after_load (bool): Whether to delete the local copy after loading (default: False)
                               Only applies when loading from MinIO storage
            init_without_load (bool): If True, skip loading the model from local or MinIO (default: True)
                                In this case, the bucket will be created to store the model if it does not exist
            overwrite_minio_bucket (bool): If True, overwrite the existing bucket in MinIO (default: False)
        
        Raises:
            AssertionError: If neither local_path nor (storage and bucket_name) are provided
            ValueError: If BM25 state dictionary cannot be loaded
        """
        self.analyzer = build_default_analyzer(language=language)
        self.bm25 = BM25EmbeddingFunction(analyzer=self.analyzer)
        self.storage_client = storage
        self.bucket_name = bucket_name

        if init_without_load:
            # Check if the bucket exists and create it if not
            if not self.storage_client or not self.bucket_name:
                logger.warning("Storage client or bucket name not provided - skipping creation of bucket...")
                return

            # Check if the bucket exists and create it if not
            if not self.storage_client.bucket_exists(bucket_name):
                logger.info(f"Bucket '{bucket_name}' does not exist. Creating a new bucket...")
                self.storage_client.make_bucket(bucket_name)
                logger.info(f"Created bucket '{bucket_name}'")
                return
            
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
                if not os.path.exists(local_path):
                    raise ValueError(f"BM25 state dict not found at {local_path}")

                logger.info(f"Loading BM25 state dict from local path: {local_path}")
                self.bm25.load(local_path)
            elif self.bucket_name and self.storage_client:
                if not os.path.exists("./bm25_state_dict.json"):
                    logger.info("Downloading the BM25 state dict...")

                    # Download the state dict from MinIO
                    retry = 3
                    while not os.path.exists("./bm25_state_dict.json"):
                        try:
                            self.storage_client.fget_object(
                                bucket_name=self.bucket_name,
                                object_name="bm25/state_dict.json",
                                file_path="./bm25_state_dict.json"
                            )
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
                raise ValueError("Please provide the local path to the BM25 state dict or the storage client and bucket name")
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
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the BM25 embeddings."""
        return self.bm25.dim
    
    @staticmethod
    def dict_to_csr(state_dict: Dict[int, float], dim: int) -> csr_array:
        """
        Convert a dictionary to a sparse CSR array.
        
        Args:
            state_dict (Dict[int, float]): Dictionary with integer keys and float values
            dim (int): Dimension of the sparse array

        Returns:
            csr_array: Sparse CSR array representation of the dictionary
        """
        cols = np.fromiter(state_dict.keys(), dtype=int)
        data = np.fromiter(state_dict.values(), dtype=float)
        rows = np.zeros(len(cols), dtype=int)
        return csr_array((data, (rows, cols)), shape=(1, dim))
    
    @staticmethod
    def dicts_to_csrs_parallel(
        state_dicts: List[Dict[int, float]], 
        dim: int, 
        max_workers: Optional[int] = None
    ) -> List[SparseEmbedding]:
        """
        Convert a list of dictionaries to a list of sparse CSR arrays in parallel.

        Args:
            state_dicts (List[Dict[int, float]]): List of dictionaries to convert.
            dim (int): Dimension of the sparse arrays.
            max_workers (Optional[int]): Maximum number of worker threads. 
                                         Defaults to min(32, os.cpu_count() + 4).

        Returns:
            List[SparseEmbedding]: List of sparse CSR array representations.
        """
        if not state_dicts:
            return []

        # Determine the number of workers
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        # Ensure we don't use more workers than tasks if the list is small
        max_workers = min(max_workers, len(state_dicts))

        results = [None] * len(state_dicts) # Pre-allocate list for potential minor efficiency

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks and store futures
            futures = {executor.submit(BM25Client.dict_to_csr, state_dict, dim): i 
                       for i, state_dict in enumerate(state_dicts)}
            
            # Process completed futures as they finish
            for future in tqdm(futures.keys(), total=len(state_dicts), desc="Converting dicts to CSR"):
                try:
                    result = future.result()
                    index = futures[future]
                    results[index] = result
                except Exception as e:
                    index = futures[future]
                    logger.error(f"Error converting dictionary at index {index}: {e}")
                    # Optionally handle error, e.g., results[index] = None or raise

        # Filter out potential None results if errors were handled by setting None
        # return [res for res in results if res is not None]
        # Or if errors should halt execution, the exception would have been raised already.
        return results
    
    @staticmethod
    def csr_to_dict(csr: csr_array | csr_matrix) -> Dict[int, float]:
            """
            Convert a sparse CSR array or matrix to a dictionary {col_index: value}.
            
            Args:
                csr (csr_array | csr_matrix): Sparse CSR array or matrix to convert

            Returns:
                state_dict (Dict[int, float]): Dictionary representation of the sparse array
            """
            mat = csr.tocsr()
            if mat.shape[0] != 1:
                raise ValueError("Input phải có đúng 1 hàng")
            start, end = mat.indptr[0], mat.indptr[1]
            cols = mat.indices[start:end]
            vals = mat.data[start:end]
            return dict(zip(cols.tolist(), vals.tolist()))
    
    @staticmethod
    def calculate_similarity(emb1: csr_array, emb2: csr_array) -> float:
        """
        Calculate the cosine similarity between two sparse CSR arrays.
        
        Args:
            emb1 (csr_array): First sparse array
            emb2 (csr_array): Second sparse array

        Returns:
            float: Cosine similarity between the two arrays
        """
        # Ensure both embeddings are in CSR format
        if not isinstance(emb1, csr_array) or not isinstance(emb2, csr_array):
            raise ValueError("Both embeddings must be in CSR format")
        
        # Calculate the dot product and norms
        dot_product = emb1.dot(emb2.T).data[0]
        norm1 = np.sqrt(emb1.power(2).sum())
        norm2 = np.sqrt(emb2.power(2).sum())
        
        # Calculate cosine similarity
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

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

        # Determine if we need to save locally
        has_storage = self.storage_client and self.bucket_name
        need_local_save = auto_save_local or has_storage  # Save locally if keeping or need to upload
        
        if need_local_save:
            # Ensure directory exists and save locally
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self.bm25.save(path)
            logger.info(f"Saved BM25 state dict to {path}")
        
        # Upload to MinIO if storage is configured
        if has_storage:
            logger.info(f"Uploading BM25 state dict to MinIO bucket '{self.bucket_name}'")
            self.storage_client.fput_object(
                bucket_name=self.bucket_name,
                object_name="bm25/state_dict.json",
                file_path=path
            )
            logger.info(f"BM25 state dict uploaded to MinIO bucket '{self.bucket_name}'")
            
            # Remove local file if auto_save_local is False (after successful upload)
            if not auto_save_local and os.path.exists(path):
                os.remove(path)
                logger.info(f"Local BM25 state dict removed from {path}")
        else:
            if not auto_save_local:
                logger.info("No storage configured and auto_save_local=False - BM25 model trained but not saved")
            else:
                logger.info(f"BM25 state dict saved locally at {path}")

    def fit_transform(self, data: List[str], path: str = "./bm25_state_dict.json", auto_save_local: bool = False) -> List[SparseEmbedding]:
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
    
    def _encode_documents(self, documents: List[str]) -> csr_array:
        """
        Convert text documents to BM25 embeddings in parallel using ThreadPool.
        
        Args:
            documents (List[str]): List of text documents to encode
            
        Returns:
            csr_array: Stacked sparse array of BM25 embeddings
        """
        # Calculate optimal number of workers - using CPU count or a fixed number

        if len(documents) < 4: 
            sparse_embs = [self.bm25._encode_document(doc) for doc in documents]
        # Process documents in parallel
        else:
            with ThreadPoolExecutor(max_workers=12) as executor:
                # Map the encoding function to each document
                sparse_embs = list(executor.map(self.bm25._encode_document, documents))
            
        # Stack the sparse embeddings and convert to CSR format
        return vstack(sparse_embs).tocsr()

    def encode_text(self, data: List[str]) -> List[SparseEmbedding]:
        """
        Convert text documents to BM25 embeddings.
        
        Args:
            data (List[str]): List of text documents to encode
            
        Returns:
            List[SparseEmbedding]: List of BM25 embeddings for the input documents
        """
        return list(self._encode_documents(data))

    def encode_queries(self, queries: List[str]) -> List[SparseEmbedding]:
        """
        Convert search queries to BM25 embeddings.
        
        Args:
            queries (List[str]): List of search queries to encode
            
        Returns:
            List[SparseEmbedding]: List of BM25 embeddings for the input queries
        """
        return list(self.bm25.encode_queries(queries))
    