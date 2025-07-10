from typing import Dict, List, Optional 

import asyncio
import time
import traceback
import lancedb
import pandas as pd
import pyarrow as pa
from datetime import timedelta
from lancedb.rerankers import RRFReranker
from loguru import logger
from minio import Minio, S3Error

from chatbot.utils.database_clients.base_class import EmbeddingData, EmbeddingType
from chatbot.utils.database_clients.base_vector_database import BaseVectorDatabase
from chatbot.utils.database_clients.lancedb.config import HostType, LanceDBConfig
from chatbot.utils.database_clients.lancedb.exceptions import (
    ConnectLanceDBDatabaseError,
    InsertLanceDBVectorError
)
from chatbot.utils.database_clients.lancedb.utils import (
    CreateMode,
    DataType,
    ElementType,
    MetricType,
    QueryType,
    ScalarIndexType,
    SchemaField,
    VectorIndexType,
)


class LanceDBVectorDatabase(BaseVectorDatabase):
    """
    LanceDB implementation for vector database.
    
    Example
    -------

    Initialize the LanceDBVectorDatabase with configuration:
    >>> config = LanceDBConfig(
            host="localhost",
            port="9000",
            bucket_name="my_lancedb_bucket",
            prefix="my_prefix",
            aws_access_key_id="your_aws_access_key_id",
            aws_secret_access_key="your_aws_secret_access_key",
            secure=False
        )
    >>> db = LanceDBVectorDatabase(config)

    Initialize the async client:
    >>> await db._initialize_async_client()
    """

    def _initialize_client(self, **kwargs) -> None:
        """Initialize the LanceDB client."""
        config: LanceDBConfig = self.config
        self.reranker = RRFReranker()
        try:
            if config.local_path and config.host_type == HostType.LOCAL:
                # If a local path is provided, use it for LanceDB
                self.client = lancedb.connect(uri=config.local_path)
                logger.info(f"Connected to LanceDB at {config.local_path} using local storage.")
                return
            
            if config.cloud_uri and config.host_type == HostType.CLOUD:
                # If a cloud URI is provided, use it for LanceDB
                self.client = lancedb.connect(
                    uri=config.cloud_uri,
                    api_key= config.api_key,
                    region=config.region,
                )
                logger.info(f"Connected to LanceDB at {config.cloud_uri} using cloud storage.")
                return
            
            if config.host_type == HostType.MINIO:

                if (
                    not config.bucket_name 
                    or not config.aws_access_key_id
                    or not config.aws_secret_access_key
                    or not config.endpoint
                    or not config.prefix
                    or not config.bucket_path
                ):
                    logger.error("Missing MinIO configuration.")
                    raise ValueError("Missing MinIO configuration.")

                # Check if the bucket exists and create it if not
                storage_client = Minio(
                    endpoint=config.endpoint.replace("http://", "").replace("https://", ""),
                    access_key=config.aws_access_key_id,
                    secret_key=config.aws_secret_access_key,
                    secure=config.secure
                )
                if not storage_client.bucket_exists(config.bucket_name):
                    storage_client.make_bucket(config.bucket_name)
                    logger.info(f"Bucket '{config.bucket_name}' created successfully.")
                else:
                    logger.info(f"Bucket '{config.bucket_name}' already exists.")
            
                # Check and create prefix folder if specified
                if config.prefix:
                    prefix_path = f"{config.prefix}/"
                    try:
                        # Try to list objects with the prefix to check if folder exists
                        objects = list(storage_client.list_objects(
                            config.bucket_name, 
                            prefix=prefix_path, 
                            recursive=False
                        ))
                        
                        # If no objects found with this prefix, create a placeholder object
                        if not objects:
                            # Create an empty object to represent the folder
                            from io import BytesIO
                            empty_data = BytesIO(b"")
                            storage_client.put_object(
                                config.bucket_name,
                                f"{prefix_path}.lancedb_folder",  # Hidden marker file
                                empty_data,
                                length=0
                            )
                            logger.info(f"Prefix folder '{config.prefix}' created in bucket '{config.bucket_name}'.")
                        else:
                            logger.info(f"Prefix folder '{config.prefix}' already exists in bucket '{config.bucket_name}'.")
                            
                    except S3Error as e:
                        logger.warning(f"Could not check/create prefix folder: {e}")

                # Connect to LanceDB
                logger.info(f"Connecting to LanceDB at {config.bucket_path}...")
                self.client = lancedb.connect(
                    uri=config.bucket_path,
                    storage_options={
                        "endpoint": config.endpoint,
                        "aws_access_key_id": config.aws_access_key_id,
                        "aws_secret_access_key": config.aws_secret_access_key,
                        "allow_http": "true" if config.secure is False else "false"
                    }
                )
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            logger.debug(traceback.format_exc())
            raise ConnectLanceDBDatabaseError(f"Failed to connect to LanceDB: {e}")

    async def _initialize_async_client(self, **kwargs) -> None:
        """Initialize the LanceDB async client."""
        config: LanceDBConfig = self.config
        try:
            if config.local_path and config.host_type == HostType.LOCAL:
                # If a local path is provided, use it for LanceDB
                self.async_client = await lancedb.connect_async(uri=config.local_path)
                logger.info(f"Connected to LanceDB at {config.local_path} using local storage.")
                return
            
            if config.cloud_uri and config.host_type == HostType.CLOUD:
                # If a cloud URI is provided, use it for LanceDB
                self.async_client = await lancedb.connect_async(
                    uri=config.cloud_uri,
                    api_key=config.api_key,
                    region=config.region,
                )
                logger.info(f"Connected to LanceDB at {config.cloud_uri} using cloud storage.")
                return
            
            if config.host_type == HostType.MINIO:
                if (
                    not config.bucket_name
                    or not config.aws_access_key_id
                    or not config.aws_secret_access_key
                    or not config.endpoint
                    or not config.prefix
                    or not config.bucket_path
                ):
                    logger.error("Missing MinIO configuration.")
                    raise ValueError("Missing MinIO configuration.")

                self.async_client = await lancedb.connect_async(
                    uri=config.bucket_path,
                    storage_options={
                        "endpoint": config.endpoint,
                        "aws_access_key_id": config.aws_access_key_id,
                        "aws_secret_access_key": config.aws_secret_access_key,
                        "allow_http": "true" if config.secure is False else "false"
                    }
                )
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB asynchronously: {e}")
            logger.debug(traceback.format_exc())
            raise ConnectLanceDBDatabaseError(f"Failed to connect to LanceDB asynchronously: {e}")

    def wait_for_index(self, index_name: str):
        POLL_INTERVAL = 10
        while True:
            indices = self.table.list_indices()

            if indices and any(index.name == index_name for index in indices):
                break
            logger.info(f"⏳ Waiting for {index_name} to be ready...")
            time.sleep(POLL_INTERVAL)

        logger.info(f"✅ Index `{index_name}` is ready!")

    def log_index_stats(self, index_names: List[str]):
        log_data = [
            "📊 Index stats for the collection:"
        ]
        for index_name in index_names:
            stat_result = self.table.index_stats(index_name)
            log_data.append((
                f"[+] Index stats for field '{index_name}': {stat_result.num_indexed_rows} indexed rows, "
                f"{stat_result.num_unindexed_rows} unindexed rows."
            ))
        logger.info("\n" + "\n".join(log_data))

    def _create_schema(
        self,
        collection_structure: List[SchemaField],
    ) -> pa.Schema:
        """
        Create a schema for the collection based on the provided structure.
        This method converts the collection structure into a PyArrow schema.

        Args:
            collection_structure (List[SchemaField]): List of SchemaField objects containing the field structure.

        Returns:
            pa.Schema: The PyArrow schema representing the collection structure.
        """

        fields = []
        for field in collection_structure:
            field_name = field.field_name
            field_type = field.field_type

            if field_type == DataType.INT:
                schema_field = pa.field(field_name, pa.int64())
            elif field_type == DataType.FLOAT:
                schema_field = pa.field(field_name, pa.float64())
            elif field_type == DataType.VECTOR:
                schema_field = pa.field(
                    field_name,
                    pa.list_(pa.float32(), field.dimension)
                )
            elif field_type == DataType.STRING:
                schema_field = pa.field(field_name, pa.string())
            elif field_type == DataType.ARRAY:
                element_type = field.element_type or ElementType.STRING
                if element_type == ElementType.STRING:
                    schema_field = pa.field(field_name, pa.list_(pa.string()))
                elif element_type == ElementType.INT:
                    schema_field = pa.field(field_name, pa.list_(pa.int64()))
                elif element_type == ElementType.FLOAT:
                    schema_field = pa.field(field_name, pa.list_(pa.float64()))
                elif element_type == ElementType.BOOL:
                    schema_field = pa.field(field_name, pa.list_(pa.bool_()))
            elif field_type == DataType.BOOL:
                schema_field = pa.field(field_name, pa.bool_())
            elif field_type == DataType.BINARY:
                schema_field = pa.field(
                    field_name,
                    pa.list_(pa.uint8(), field.dimension)
                )
            else:
                logger.warning(f"Unsupported field type '{field_type}' for field '{field_name}'. Skipping.")

            fields.append(schema_field)

        return pa.schema(fields)
    
    def _create_index(
        self,
        collection_structure: List[SchemaField]
    ) -> bool:
        """
        Create an index for the collection based on the provided structure.
        This method creates a vector index for the specified fields in the collection.

        Args:
            collection_structure (List[SchemaField]): List of SchemaField objects containing the collection structure.

        Returns:
            bool: True if the index was created successfully, False otherwise.
        """
        try:
            # Create index for each field in the collection structure
            for field in collection_structure:
                # Skip fields that do not require indexing
                index_config = field.index_config
                if not index_config.index:
                    continue

                if field.field_type == DataType.VECTOR:
                    if self.config.host_type == HostType.CLOUD:
                        # For cloud, use IVF_HNSW_SQ index type
                        self.table.create_index(
                            vector_column_name=field.field_name,
                            metric=MetricType.COSINE.value,
                        )
                    else:
                        self.table.create_index(
                            vector_column_name=field.field_name,
                            index_type=VectorIndexType.IVF_HNSW_PQ.value,
                            metric=MetricType.COSINE.value,
                            ef_construction=300,
                            m=16, # nprobes
                            replace=True,
                        )
                elif field.field_type == DataType.BINARY:
                    self.table.create_index(
                        vector_column_name=field.field_name,
                        index_type=VectorIndexType.IVF_FLAT.value,
                        metric=MetricType.HAMMING.value,
                        replace=True,
                    )
                elif field.field_type == DataType.ARRAY:
                    if field.element_type == ElementType.STRING:
                        self.table.create_scalar_index(
                            column=field.field_name,
                            index_type=ScalarIndexType.LABEL_LIST.value,
                            replace=True,
                        )
                elif field.field_type in [DataType.INT, DataType.FLOAT, DataType.STRING]:
                    if index_config.full_text_search and field.field_type == DataType.STRING:
                        if self.config.host_type == HostType.CLOUD:
                            self.table.create_fts_index(field.field_name)
                        else:
                            self.table.create_fts_index(
                                field_names=field.field_name,
                                # Tantivy is not supported for storing
                                # in object storage like MinIO
                                use_tantivy=False,
                                replace=True,
                            )
                    else:
                        self.table.create_scalar_index(
                            column=field.field_name,
                            index_type=ScalarIndexType.BTREE.value,
                            replace=True,
                        )

                self.wait_for_index(field.field_name + "_idx")

            # Log the index stats for all indices in the collection
            index_names = [field.field_name + "_idx" for field in collection_structure if field.index_config.index]
            self.log_index_stats(index_names)

            logger.info("📢 📢 📢 Index created successfully for the collection.")
            return True
        except Exception as e:
            logger.error(f"Failed to create index for the collection: {e}")
            logger.debug(traceback.format_exc())
            return False

    def create_collection(
        self,
        collection_name: str,
        collection_structure: List[SchemaField],
        data: pd.DataFrame,
        **kwargs,
    ) -> None:
        """
        Create a new collection in the vector database.
        This method checks if the collection already exists, and if so, it overwrites it.
        If the collection does not exist, it creates a new one with the specified structure and data.
        This method also creates the necessary index for the collection based on the provided structure.

        Args:
            collection_name (str): The name of the collection to be created.
            collection_structure (List[SchemaField]): The structure of the collection, defining the fields and their types.
            data (pd.DataFrame): The data to be inserted into the collection.
        """
        # Check if the collection already exists
        overwrite_collection = False
        try:
            self.client[collection_name]
            overwrite_collection = True
        except Exception as e:
            # Continue to create the collection if it does not exist
            pass

        if overwrite_collection:
            logger.warning((
                f"Collection '{collection_name}' already exists. "
                "The collection will be overwritten."
            ))

        # Create the collection schema and load the data
        schema = self._create_schema(collection_structure)

        self.table = self.client.create_table(
            name=collection_name,
            data=data,
            schema=schema,
            mode=(
                CreateMode.OVERWRITE.value 
                if overwrite_collection else CreateMode.CREATE.value
            )
        )

        # Create the index for the collection
        if not self._create_index(collection_structure):
            logger.error(f"Failed to create index for collection '{collection_name}'.")
            raise Exception(f"Failed to create index for collection '{collection_name}'.")

        logger.info(f"Collection '{collection_name}' created successfully.")

    def load_collection(self, collection_name: str, **kwargs) -> bool:
        """Load the collection into memory for faster search operations."""
        try:
            self.table = self.client.open_table(collection_name)
            logger.info(f"Collection '{collection_name}' loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection '{collection_name}': {e}")
            logger.debug(traceback.format_exc())
            return False

    def delete_collection(self, collection_name: str, **kwargs) -> None:
        """Delete a collection from the database."""
        pass
    
    def list_collections(self, **kwargs) -> List[str]:
        """List all collections in the database."""
        pass

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        """Check if a collection exists."""
        pass

    def insert_vectors(self, data: List[Dict], collection_structure: Optional[List[SchemaField]], **kwargs) -> None:
        """Insert vectors into the collection."""
        # Insert data into the collection
        try:
            logger.info(f"Inserting data into collection '{self.table.name}'...")
            self.table.add(data)

            if self.config.host_type != HostType.CLOUD:
                # Re-optimize the index after inserting data
                logger.info(f"Re-optimizing the index for collection '{self.table.name}' after inserting data...")
                try:
                    self.table.optimize()
                except:
                    # If optimize fails, drop the index and recreate it
                    logger.warning((
                        f"Failed to optimize index for collection '{self.table.name}'. "
                        "Dropping and recreating the index."
                    ))

                    if not collection_structure:
                        raise ValueError(
                            "Collection structure must be provided to recreate the index."
                        )

                    self._create_index(collection_structure)

                    # Clean old versions of the table
                    self.table.cleanup_old_versions(timedelta(days=7))
            else:
                # For cloud storage, no need to manually re-optimize the index
                logger.info("No need to manually re-optimize the index for cloud storage.")

            logger.info(f"Data inserted successfully into collection '{self.table.name}'.")
        except Exception as e:
            logger.error(f"Failed to insert data into collection '{self.table.name}': {e}")
            logger.debug(traceback.format_exc())
            raise InsertLanceDBVectorError(f"Failed to insert data into collection '{self.table.name}': {e}")

    def get_items(self, ids: List[str] = [], id_field: str = None, **kwargs) -> List[Dict]:
        """Get items from the collection by their IDs."""
        if not id_field:
            raise ValueError("id_field must be specified to retrieve items by ID.")

        # Convert the LanceDB table to a pandas DataFrame
        df = self.table.to_pandas()

        # If no IDs are provided, return all records
        if not ids:
            return df.to_dict(orient="records")
        
        return df[df[id_field].isin(ids)].to_dict(orient="records")

    def search_dense_vectors(
        self,
        query_embeddings: List[List[float]],
        field_name: str,
        output_fields: List[str],
        top_k: int,
        filtering_expr: str = "",
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Search for dense vectors in the collection.
        
        Args:
            query_embeddings (List[List[float]]): List of query embeddings to search for.
            field_name (str): The name of the field containing the vector data.
            output_fields (List[str]): List of fields to return in the search results.
            filtering_expr (str): Filtering expression to apply to the search results.
            top_k (int): Number of top results to return.

        Returns:
            List[List[Dict]]: List of top-k search results of each input query embedding, 
                containing the specified output fields.
                The number of lists in the output is equal to the number of query embeddings.
        """
        results = []
        for query_embedding in query_embeddings:
            # Perform vector search for each query embedding
            result = (
                self.table.search(
                    query=query_embedding,
                    vector_column_name=field_name,
                    query_type=QueryType.VECTOR.value,
                ).where(filtering_expr, prefilter=True)
                .select(output_fields)
                .limit(top_k)
                .to_list()
            )

            # Convert _distance to _score for each result
            for item in result:
                if "_distance" in item:
                    item["_score"] = item.pop("_distance")

            # Append the result to the results list
            results.append(result)

        return results

    def hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int,
        **kwargs,
    ) -> List[Dict]:
        """
        Perform hybrid search for vectors in the collection.

        Args:
            embedding_data (List[EmbeddingData]): List of EmbeddingData objects containing the vector data and metadata.
            output_fields (List[str]): List of fields to return in the search results.
            top_k (int): Number of top results to return.
            filtering_expr (str): Filtering expression to apply to the search results.

        Returns:
            result (List[Dict]): List of searched items, each containing the specified output fields and key `_score` key.

        Example
        -------

        >>> embedding_data = [
            EmbeddingData(
                vector=[0.1, 0.2, 0.3],
                embedding_type=EmbeddingType.DENSE,
                field_name="vector_field"
            ),
            EmbeddingData(
                query="this is a text query",
                embedding_type=EmbeddingType.SPARSE,
                field_name="text_field",
            )
        ]
        >>> output_fields = ["id", "name"]
        >>> top_k = 5
        >>> filtering_expr = "category == 'food'"
        >>> results = db.hybrid_search_vectors(
                embedding_data=embedding_data,
                output_fields=output_fields,
                top_k=top_k
            )

            [
                {"id": "1", "name": "Pizza", "_score": 0.95},
                {"id": "2", "name": "Burger", "_score": 0.90},
                ...
            ]
        """
        results = []
        for embedding in embedding_data:
            if not isinstance(embedding, EmbeddingData):
                raise TypeError("Invalid embedding data type. Expected EmbeddingData.")
            
            embedding_type = embedding.embedding_type
            field_name = embedding.field_name
            filtering_expr = embedding.filtering_expr or ""
            if embedding_type == EmbeddingType.SPARSE:
                result = (
                    self.table.search(
                        query=embedding.query,
                        fts_columns=field_name,
                        query_type=QueryType.FTS.value,
                    ).where(filtering_expr, prefilter=True)
                    .select(output_fields)
                    .with_row_id(True)  # Ensure _rowid is included for reranking
                    .limit(top_k)
                    .to_arrow()
                )
            elif embedding_type in [EmbeddingType.DENSE, EmbeddingType.BINARY]:
                result = (
                    self.table.search(
                        query=embedding.embeddings,
                        vector_column_name=field_name,
                        query_type=QueryType.VECTOR.value,
                    )
                    .where(filtering_expr, prefilter=True)
                    .select(output_fields)
                    .with_row_id(True)  # Ensure _rowid is included for reranking
                    .limit(top_k)
                )

            results.append(result)

        final_results: pa.Table = (
            self.reranker.rerank_multivector(results)
            .to_pylist()
        )[:top_k]

        # Convert _relevance_score to _score for each result
        for item in final_results:
            if "_relevance_score" in item:
                item["_score"] = item.pop("_relevance_score")

        return final_results

    def async_create_collection(
        self,
        collection_name: str,
        collection_structure: List[Dict],
        **kwargs
    ) -> None:
        """Asynchronously create a new collection in the vector database."""
        pass

    async def async_load_collection(self, collection_name: str, **kwargs) -> bool:
        """Asynchronously load the collection into memory for faster search operations."""
        try:
            self.async_table = await self.async_client.open_table(collection_name)
            logger.info(f"Collection '{collection_name}' loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection '{collection_name}': {e}")
            logger.debug(traceback.format_exc())
            return False

    async def async_delete_collection(self, collection_name: str, **kwargs) -> None:
        """Asynchronously delete a collection from the database."""
        pass

    async def async_list_collections(self, **kwargs) -> List[str]:
        """Asynchronously list all collections in the database."""
        pass

    async def async_has_collection(self, collection_name: str, **kwargs) -> bool:
        """Asynchronously check if a collection exists."""
        pass

    async def async_insert_vectors(
        self, data: List[Dict], **kwargs
    ) -> None:
        """Asynchronous insert vectors into the collection."""
        pass

    async def async_get_items(self, ids: List[str] = [], id_field: str = None, **kwargs) -> List[Dict]:
        """Asynchronously get items from the collection by their IDs."""
        return await asyncio.to_thread(
            self.get_items, ids=ids, id_field=id_field, **kwargs
        )
        
    async def async_search_dense_vectors(
        self,
        query_embeddings: List[List[float]],
        field_name: str,
        output_fields: List[str],
        top_k: int,
        filtering_expr: str = "",
        **kwargs
    ) -> List[List[Dict]]:
        """
        Asynchronously search for dense vectors in the collection.
        
        Args:
            query_embeddings (List[List[float]]): List of query embeddings to search for.
            field_name (str): The name of the field containing the vector data.
            output_fields (List[str]): List of fields to return in the search results.
            filtering_expr (str): Filtering expression to apply to the search results.
            top_k (int): Number of top results to return.

        Returns:
            List[List[Dict]]: List of top-k search results of each input query embedding, 
                containing the specified output fields.
                The number of lists in the output is equal to the number of query embeddings.
        """
        results = []
        for query_embedding in query_embeddings:
            # Perform vector search for each query embedding
            if filtering_expr:
                result = await (
                    self.async_table.vector_search(query_vector=query_embedding)
                    .column(column=field_name)
                    .fast_search()
                    .where(predicate=filtering_expr)
                    .select(columns=output_fields)
                    .limit(limit=top_k)
                    .to_list()
                )
            else:
                result = await (
                    self.async_table.vector_search(query_vector=query_embedding)
                    .column(column=field_name)
                    .fast_search()
                    .select(columns=output_fields)
                    .limit(limit=top_k)
                    .to_list()
                )

            # Convert _distance to _score for each result
            for item in result:
                if "_distance" in item:
                    item["_score"] = item.pop("_distance")

            # Append the result to the results list
            results.append(result)

        return results

    async def async_hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int,
        **kwargs
    ) -> List[Dict]:
        """
        Asynchronously perform hybrid search for vectors in the collection.

        Args:
            embedding_data (List[EmbeddingData]): List of EmbeddingData objects containing the vector data and metadata.
            output_fields (List[str]): List of fields to return in the search results.
            top_k (int): Number of top results to return.
            filtering_expr (str): Filtering expression to apply to the search results.

        Returns:
            result (List[Dict]): List of searched items, each containing the specified output fields and key `_score` key.

        Example
        -------

        >>> embedding_data = [
            EmbeddingData(
                vector=[0.1, 0.2, 0.3],
                embedding_type=EmbeddingType.DENSE,
                field_name="vector_field"
            ),
            EmbeddingData(
                query="this is a text query",
                embedding_type=EmbeddingType.SPARSE,
                field_name="text_field",
            )
        ]
        >>> output_fields = ["id", "name"]
        >>> top_k = 5
        >>> filtering_expr = "category == 'food'"
        >>> results = await db.async_hybrid_search_vectors(
                embedding_data=embedding_data,
                output_fields=output_fields,
                top_k=top_k
            )

            [
                {"id": "1", "name": "Pizza", "_score": 0.95},
                {"id": "2", "name": "Burger", "_score": 0.90},
                ...
            ]
        """
        return await asyncio.to_thread(
            self.hybrid_search_vectors,
            embedding_data=embedding_data,
            output_fields=output_fields,
            top_k=top_k,
            **kwargs
        )
    