from typing import List, Dict

import asyncio
import traceback
from loguru import logger
from pymilvus import (
    connections,
    AnnSearchRequest,
    AsyncMilvusClient,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker
)
from pymilvus.milvus_client import IndexParams

from chatbot.utils.database_clients.base_class import EmbeddingData, EmbeddingType
from chatbot.utils.database_clients.base_vector_database import BaseVectorDatabase
from chatbot.utils.database_clients.milvus.config import MilvusConfig
from chatbot.utils.database_clients.milvus.exceptions import (
    CreateMilvusCollectionError,
    InsertMilvusVectorsError,
    GetMilvusItemsError,
    SearchMilvusVectorsError
)
from chatbot.utils.database_clients.milvus.utils import IndexParam, IndexType, MetricType, SchemaField


class MilvusVectorDatabase(BaseVectorDatabase):
    """
    Milvus implementation for vector database.
    
    Example
    ------

    Initialize the Milvus database client connecting to a local Milvus server:

    >>> config = MilvusConfig(host="localhost", port="19530", run_async=False)
    >>> milvus_db = MilvusVectorDatabase(config=config)

    Initialize the Milvus database client connecting to a cloud Milvus server:

    >>> config = MilvusConfig(
    ...     cloud_uri="https://your-cloud-milvus-uri",
    ...     token="your_token",
    ...     run_async=False
    ... )
    >>> milvus_db = MilvusVectorDatabase(config=config)

    Load collection into memory

    >>> milvus_db.load_collection("your_collection_name")

    """

    def _initialize_client(self) -> None:
        """Initialize the Milvus client."""
        config: MilvusConfig = self.config
        self.client = MilvusClient(uri=config.uri, token=config.token)
        self.async_client = AsyncMilvusClient(uri=config.uri, token=config.token) if config.run_async else None
        self.reranker = RRFRanker()
        self.run_async = config.run_async

    def _create_schema_and_index(
        self,
        collection_structure: List[SchemaField],
        auto_id: bool = False,
        enable_dynamic_field: bool = False,
        json_index_params: Dict[str, List[IndexParam]] = None
    ) -> tuple[CollectionSchema, IndexParams]:
        """
        Create schema and index parameters for a collection.

        Args:
            collection_structure (List[SchemaField]): List of SchemaField objects defining the structure of the collection.
            auto_id (bool): Enable auto ID generation for the collection.
            enable_dynamic_field (bool): Enable dynamic field for the collection allowing new fields to be added.
            json_index_params (Dict[str, List[IndexParam]]): Index parameters of JSON type for the collection.
                Key is the field name and value is a list of IndexParam objects.

        Returns:
            tuple[CollectionSchema, IndexParams]: Schema and index parameters for the collection.
        """
        data_type_mapping = {
            "int": DataType.INT64,
            "string": DataType.VARCHAR,
            "dense_vector": DataType.FLOAT_VECTOR,
            "sparse_vector": DataType.SPARSE_FLOAT_VECTOR,
            "array": DataType.ARRAY,
            "bool": DataType.BOOL,
            "json": DataType.JSON,
            "binary": DataType.BINARY_VECTOR,
            "float": DataType.FLOAT,
        }

        fields = []
        index_params = self.client.prepare_index_params()

        for field in collection_structure:
            if field.field_type.value == "int":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.STL_SORT.value,
                    )
            elif field.field_type.value == "dense_vector":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    dim=field.dimension,
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_result = self.check_index_type(field.index_config.index_type)
                    if index_result != field.index_config.index_type:
                        raise CreateMilvusCollectionError(f"Error creating collection: {index_result}")

                    metric_result = self.check_metric_type(field.index_config.metric_type)
                    if metric_result != field.index_config.metric_type:
                        raise CreateMilvusCollectionError(f"Error creating collection: {metric_result}")
                    
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=index_result.value,
                        metric_type=metric_result.value,
                        params={
                            "M": (
                                field.index_config.hnsw_m 
                                if field.index_config and field.index_config.hnsw_m 
                                else 16
                            ),
                            "efConstruction": (
                                field.index_config.hnsw_ef_construction 
                                if field.index_config and field.index_config.hnsw_ef_construction 
                                else 500
                            )
                        }
                    )
            elif field.field_type.value == "sparse_vector":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.SPARSE_INVERTED_INDEX.value,
                        metric_type=MetricType.IP.value,
                    )
            elif field.field_type.value == "string":
                analyzer_params = {"tokenizer": "standard", "filter": ["lowercase", "asciifolding"]}
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    max_length=65535,
                    analyzer_params=analyzer_params,
                    enable_analyzer=True,
                    enable_match=True,
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.INVERTED.value
                    )
            elif field.field_type.value == "array":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    element_type=data_type_mapping[field.element_type.value],
                    max_capacity=field.max_capacity or 100,
                    max_length=65535,
                    is_primary=field.is_primary,
                    nullable=True
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.AUTOINDEX.value
                    )
            elif field.field_type.value == "bool":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.AUTOINDEX.value
                    )
            elif field.field_type.value == "json":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    # Add index parameters for JSON fields
                    if json_index_params and json_index_params.get(field.field_name, None):
                        for _, index_param in enumerate(json_index_params[field.field_name]):
                            index_params.add_index(
                                field_name=field.field_name,
                                index_type=IndexType.INVERTED.value,
                                index_name=index_param.index_name,
                                params={
                                    "json_path": index_param.indexed_key,
                                    "json_cast_type": index_param.value_type.value
                                }
                            )
            elif field.field_type.value == "binary":
                schema_field = FieldSchema(
                    name=field.field_name,
                    dtype=data_type_mapping[field.field_type.value],
                    description=field.field_description or "",
                    dim=field.dimension,
                    is_primary=field.is_primary
                )

                if field.index_config and field.index_config.index:
                    index_params.add_index(
                        field_name=field.field_name,
                        index_type=IndexType.BIN_FLAT.value,
                        metric_type=MetricType.HAMMING.value,
                    )
            else:
                raise ValueError((
                    "Invalid field type. Please provide one of 'int', 'string', "
                    "'dense_vector', 'sparse_vector', 'array', 'bool', 'json', or 'binary'."
                ))
            
            fields.append(schema_field)
        
        schema = CollectionSchema(
            fields=fields,
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field
        )

        return schema, index_params

    def create_collection(
        self,
        collection_name: str,
        collection_structure: List[SchemaField],
        auto_id: bool = False,
        enable_dynamic_field: bool = False,
        json_index_params: Dict[str, List[IndexParam]] = None,
        **kwargs,
    ):
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): Name of the collection to create.
            collection_structure (List[SchemaField]): List of SchemaField objects defining the collection structure.
            auto_id (bool): Enable auto ID generation for the collection.
            enable_dynamic_field (bool): Enable dynamic field for the collection allowing new fields to be added.
            json_index_params (Dict[str, List[IndexParam]]): Index parameters of JSON type for the collection.
                Key is the field name and value is a list of IndexParam objects.
        """
        # Check if collection exists
        if self.has_collection(collection_name):
            self.delete_collection(collection_name)

        # Create schema and index parameters
        try:
            schema, index_params = self._create_schema_and_index(
                collection_structure=collection_structure,
                auto_id=auto_id,
                enable_dynamic_field=enable_dynamic_field,
                json_index_params=json_index_params
            )
        except Exception as e:
            raise CreateMilvusCollectionError(f"Error creating collection schema: {str(e)}")

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"Collection {collection_name} created successfully!")

    async def async_create_collection(
        self,
        collection_name: str,
        collection_structure: List[SchemaField],
        auto_id: bool = False,
        enable_dynamic_field: bool = False,
        json_index_params: Dict[str, List[IndexParam]] = None,
        **kwargs,
    ):
        """
        Asynchronously create a new collection in the vector database.

        Args:
            collection_name (str): Name of the collection to create.
            collection_structure (List[SchemaField]): List of SchemaField objects defining the collection structure.
            auto_id (bool): Enable auto ID generation for the collection.
            enable_dynamic_field (bool): Enable dynamic field for the collection allowing new fields to be added.
            json_index_params (Dict[str, List[IndexParam]]): Index parameters of JSON type for the collection.
                Key is the field name and value is a list of IndexParam objects.
        """
        # Check if collection exists
        if await self.async_has_collection(collection_name):
            await self.async_delete_collection(collection_name)

        # Create schema and index parameters
        try:
            schema, index_params = self._create_schema_and_index(
                collection_structure=collection_structure,
                auto_id=auto_id,
                enable_dynamic_field=enable_dynamic_field,
                json_index_params=json_index_params
            )
        except Exception as e:
            raise CreateMilvusCollectionError(f"Error creating collection schema: {str(e)}")

        # Create collection
        await self.async_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

        logger.info(f"Collection {collection_name} created successfully!")

    def load_collection(self, collection_name: str, **kwargs) -> bool:
        """
        Load the collection into memory for faster search operations.

        Args:
            collection_name (str): Name of the collection to load.

        Returns:
            bool: True if the collection is loaded successfully, False otherwise.
        """

        if not self.client.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist!")
            return False

        # Load the collection
        self.client.load_collection(collection_name)

        # Check if the collection is loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        if load_state:
            logger.info(f"Collection {collection_name} is loaded successfully!")
            return True
        else:
            logger.warning(f"Failed to load collection {collection_name}!")
            return False

    async def async_load_collection(self, collection_name: str, **kwargs) -> bool:
        """
        Asynchronously load the collection into memory for faster search operations.

        Args:
            collection_name (str): Name of the collection to load.

        Returns:
            bool: True if the collection is loaded successfully, False otherwise.
        """
        if not self.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist!")
            return False

        # Load the collection
        await self.async_client.load_collection(collection_name)

        # Check if the collection is loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        if load_state:
            logger.info(f"Collection {collection_name} is loaded successfully!")
            return True
        else:
            logger.warning(f"Failed to load collection {collection_name}!")
            return False

    def delete_collection(self, collection_name: str, **kwargs) -> None:
        """Delete a collection from Milvus."""
        self.client.drop_collection(collection_name)

    async def async_delete_collection(self, collection_name: str, **kwargs) -> None:
        """Asynchronously delete a collection from Milvus."""
        await self.async_client.drop_collection(collection_name)

    def list_collections(self, **kwargs) -> List[str]:
        """List all collections in Milvus."""
        return self.client.list_collections()

    async def async_list_collections(self, **kwargs) -> List[str]:
        """Asynchronously list all collections in Milvus."""
        return await asyncio.to_thread(self.list_collections)

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        """Check if a collection exists in Milvus."""
        return self.client.has_collection(collection_name)

    async def async_has_collection(self, collection_name: str, **kwargs) -> bool:
        """Asynchronously check if a collection exists in Milvus."""
        return await asyncio.to_thread(self.has_collection, collection_name)

    def check_metric_type(self, metric_type: MetricType) -> MetricType:
        """
        Check if the metric type is supported.
        
        Args:
            metric_type (MetricType): Metric type of the index.

        Returns:
            MetricType: Metric type if supported, error message otherwise.
        """
        assert isinstance(metric_type, MetricType), "metric_type must be an instance of MetricType Enum"

        supported_metric_types = list(MetricType)
        if metric_type not in supported_metric_types:
            return f"Invalid metric type. Please provide one of {supported_metric_types}"
        return metric_type

    def check_index_type(self, index_type: IndexType) -> IndexType:
        """
        Check if the index type is supported.
        
        Args:
            index_type (IndexType): Index type of the index.

        Returns:
            IndexType: Index type if supported, error message otherwise.
        """
        assert isinstance(index_type, IndexType), "index_type must be an instance of IndexType Enum"

        supported_index_types = list(IndexType)
        if index_type not in supported_index_types:
            return f"Invalid index type. Please provide one of {supported_index_types}"
        return index_type

    def insert_vectors(
        self,
        data: List[Dict],
        collection_name: str = None,
        **kwargs,
    ) -> None:
        """
        Insert vectors into the collection.

        Args:
            collection_name (str): Name of the collection.
            data (List[Dict]): List of dictionaries containing the data to insert.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for inserting vectors.")
        
        try:
            self.client.insert(
                collection_name=collection_name,
                data=data
            )
        except Exception as e:
            raise InsertMilvusVectorsError(f"Error inserting vectors: {str(e)}")
        
    async def async_insert_vectors(
        self,
        data: List[Dict],
        collection_name: str = None,
        **kwargs
    ) -> None:
        """
        Asynchronously insert vectors into the collection.

        Args:
            collection_name (str): Name of the collection.
            data (List[Dict]): List of dictionaries containing the data to insert.

        Raises:
            InsertMilvusVectorsError: If there is an error during insertion.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for inserting vectors.")

        try:
            await self.async_client.insert(
                collection_name=collection_name,
                data=data
            )
        except Exception as e:
            raise InsertMilvusVectorsError(f"Error inserting vectors: {str(e)}")

    def get_items(
        self,
        ids: List[str] = [],
        collection_name: str = None,
        field_names: List[str] = None,
        **kwargs
    ) -> List[dict]:
        """
        Get items from the collection by their IDs.

        Args:
            collection_name (str): Name of the collection.
            ids (List[int]): List of IDs to retrieve.

        Returns:
            List[dict]: List of dictionaries containing the items.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for getting items.")
        
        try:
            # If no IDs are provided, return all records
            if not ids:
                return self.get_all_items(collection_name, field_names)

            return self.client.get(
                collection_name=collection_name,
                ids=ids
            )
        except Exception as e:
            raise GetMilvusItemsError(f"Error getting items: {str(e)}")
        
    async def async_get_items(
        self,
        ids: List[str] = [],
        collection_name: str = None,
        field_names: List[str] = None,
        **kwargs
    ) -> List[dict]:
        """
        Asynchronously get items from the collection by their IDs.

        Args:
            collection_name (str): Name of the collection.
            ids (List[int]): List of IDs to retrieve.

        Returns:
            List[dict]: List of dictionaries containing the items.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for getting items.")

        try:
            # If no IDs are provided, return all records
            if not ids:
                return await self.async_get_all_items(collection_name, field_names)

            result = await self.async_client.get(
                collection_name=collection_name,
                ids=ids
            )
            return result
        except Exception as e:
            raise GetMilvusItemsError(f"Error getting items: {str(e)}")
    
    def get_all_items(self, collection_name: str, field_names: List[str]) -> List[dict]:
        """
        Get all items from the collection.

        Args:
            collection_name (str): Name of the collection.
            field_names (List[str]): List of fields to return in the search results.

        Returns:
            List[dict]: List of dictionaries containing all vectors and metadata.
        """
        list_records = []
        try:
            if not connections.has_connection(alias="default"):
                # If no connection exists, create a new one
                connections.connect(uri=self.config.uri, token=self.config.token, _async=self.run_async)

            # Get number of items indexed in the database
            self.collection = Collection(collection_name)
            num_items = self.collection.num_entities

            # Create an iterator to fetch items in batches
            iterator = self.client.query_iterator(
                collection_name=collection_name,
                batch_size=100,
                limit=num_items,
                output_fields=field_names
            )
            while True:
                batch = iterator.next()
                if not batch:
                    iterator.close()
                    break
                list_records.extend([record for record in batch])
            
            return list_records
        except Exception as e:
            raise GetMilvusItemsError(f"Error getting all items: {str(e)}")
        
    async def async_get_all_items(self, collection_name: str, field_names: List[str]) -> List[dict]:
        """
        Asynchronously get all items from the collection.

        Args:
            collection_name (str): Name of the collection.
            field_names (List[str]): List of fields to return in the search results.

        Returns:
            List[dict]: List of dictionaries containing all vectors and metadata.
        """
        list_records = []
        try:
            if not self.async_client.has_collection(collection_name):
                raise GetMilvusItemsError(f"Collection {collection_name} does not exist.")
            
            # Get number of items indexed in the database
            self.collection = Collection(collection_name)
            num_items = self.collection.num_entities

            # Create an iterator to fetch items in batches
            iterator = await asyncio.to_thread(
                self.async_client.query_iterator,
                collection_name=collection_name,
                batch_size=100,
                limit=num_items,
                output_fields=field_names
            )
            while True:
                batch = iterator.next()
                if not batch:
                    iterator.close()
                    break
                list_records.extend([record for record in batch])
            
            return list_records
        except Exception as e:
            raise GetMilvusItemsError(f"Error getting all items: {str(e)}")
        
    def build_hybrid_search_requests(
        self,
        embedding_data: List[EmbeddingData],
        top_k: int,
        metric_type: MetricType, 
        index_type: IndexType
    ) -> List[AnnSearchRequest]:
        """
        Build hybrid search requests for the given embedding data.

        Args:
            embedding_data (List[EmbeddingData]): List of EmbeddingData objects containing dense or/and sparse data.
            top_k (int): Number of results to return.
            metric_type (MetricType): Metric type for the dense search query.
            index_type (IndexType): Index type for the dense vector ("FLAT", "IVF_FLAT", "HNSW").

        Returns:
            List[AnnSearchRequest]: List of AnnSearchRequest objects for hybrid search.
        """
        search_requests = []
        for embedding in embedding_data:
            if not isinstance(embedding, EmbeddingData):
                raise TypeError("Invalid embedding data type. Expected EmbeddingData.")
            
            embedding_type = embedding.embedding_type
            param = {}
            if embedding_type == EmbeddingType.SPARSE:
                param = {
                    "metric_type": MetricType.IP.value,
                    "params": {}
                }
            elif embedding_type == EmbeddingType.DENSE:
                param = {
                    "metric_type": metric_type.value,
                    "params": {"ef": (top_k*2)} if index_type == IndexType.HNSW else {"nprobe": 16}
                }
            elif embedding_type == EmbeddingType.BINARY:
                param = {
                    "metric_type": MetricType.HAMMING.value,
                    "params": {"nprobe": 64}
                }
            
            search_params = {
                "data": [embedding.embeddings],
                "anns_field": embedding.field_name,
                "param": param,
                "limit": (top_k*2),
                "expr": embedding.filtering_expr
            }
            search_requests.append(AnnSearchRequest(**search_params))
        
        return search_requests
    
    def hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int = 5,
        metric_type: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        collection_name: str = None,
        **kwargs,
    ) -> List[dict]:
        """
        Perform hybrid search (in multiple types: dense or sparse or binary) for vectors in the collection.

        Args:
            collection_name (str): Name of the collection.
            embedding_data (List[EmbeddingData]): List of EmbeddingData objects containing dense or/and sparse data.
            output_fields (List[str]): List of fields to return in the search results.
            top_k (int): Number of results to return.
            metric_type (MetricType): Metric type for the dense search query.
            index_type (IndexType): Index type for dense vector.

        Returns:
            List[dict]: The top-k search result for the input query, containing the expected output fields and "_score" key.

        Example
        -------
        >>> embedding_data = [
        >>>     EmbeddingData(embedding_type=EmbeddingType.DENSE, embeddings=[[0.1, 0.2, 0.3]], field_name="dense_field"),
        >>>     EmbeddingData(embedding_type=EmbeddingType.SPARSE, embeddings=[[0.0, 1.0, 0.0]], field_name="sparse_field")
        >>> ]
        >>> output_fields = ["id", "name", "dense_field", "sparse_field"]
        >>> results = milvus_db.hybrid_search_vectors(
        >>>     collection_name="my_collection",
        >>>     embedding_data=embedding_data,
        >>>     output_fields=output_fields,
        >>>     top_k=5,
        >>>     metric_type=MetricType.COSINE,
        >>>     index_type=IndexType.HNSW
        >>> )
            [
                {
                    "id": "123",
                    "name": "example_item",
                    "dense_field": [0.1, 0.15, 0.22],
                    "sparse_field": [1.0, 1.0, 0.0],
                    "_score": 0.95
                },
                ...
            ]
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for hybrid search.")

        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {metric_result}")
        
        if not connections.has_connection(alias="default"):
            # If no connection exists, create a new one
            connections.connect(uri=self.config.uri, token=self.config.token, _async=self.run_async)

        # Construct the collection
        self.collection = Collection(collection_name)

        try:
            search_requests = self.build_hybrid_search_requests(
                embedding_data=embedding_data,
                top_k=top_k,
                metric_type=metric_type,
                index_type=index_type
            )
            if not search_requests:
                raise SearchMilvusVectorsError("No valid search requests were created. Check the input embedding data.")
            
            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=search_requests,
                ranker=self.reranker,
                limit=top_k,
                output_fields=output_fields,
                **kwargs
            )

            if not results:
                return []
                
            # Flatten the structure by moving entity fields to top level
            flattened_results = []
            for result in results[0]:
                flattened_result = {}
                
                # Copy non-entity fields (like distance, id, etc.)
                for key, value in result.items():
                    if key != "entity":
                        if key == "distance":
                            flattened_result["_score"] = value
                        else:
                            flattened_result[key] = value
                
                # Move entity fields to top level
                if "entity" in result:
                    for entity_key, entity_value in result["entity"].items():
                        flattened_result[entity_key] = entity_value
                
                flattened_results.append(flattened_result)
                
            return flattened_results
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in hybrid search: {str(e)}")
        
    async def async_hybrid_search_vectors(
        self,
        embedding_data: List[EmbeddingData],
        output_fields: List[str],
        top_k: int = 5,
        metric_type: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        collection_name: str = None,
        **kwargs,
    ) -> List[dict]:
        """
        Asynchronously perform hybrid search (in multiple types: dense or sparse or binary) for vectors in the collection.

        Args:
            collection_name (str): Name of the collection.
            embedding_data (List[EmbeddingData]): List of EmbeddingData objects containing dense or/and sparse data.
            output_fields (List[str]): List of fields to return in the search results.
            top_k (int): Number of results to return.
            metric_type (MetricType): Metric type for the dense search query.
            index_type (IndexType): Index type for dense vector.

        Returns:
            List[dict]: The top-k search result for the input query, containing the expected output fields and "_score" key.

        Example
        -------
        >>> embedding_data = [
        >>>     EmbeddingData(embedding_type=EmbeddingType.DENSE, embeddings=[[0.1, 0.2, 0.3]], field_name="dense_field"),
        >>>     EmbeddingData(embedding_type=EmbeddingType.SPARSE, embeddings=[[0.0, 1.0, 0.0]], field_name="sparse_field")
        >>> ]
        >>> output_fields = ["id", "name", "dense_field", "sparse_field"]
        >>> results = milvus_db.hybrid_search_vectors(
        >>>     collection_name="my_collection",
        >>>     embedding_data=embedding_data,
        >>>     output_fields=output_fields,
        >>>     top_k=5,
        >>>     metric_type=MetricType.COSINE,
        >>>     index_type=IndexType.HNSW
        >>> )
            [
                {
                    "id": "123",
                    "name": "example_item",
                    "dense_field": [0.1, 0.15, 0.22],
                    "sparse_field": [1.0, 1.0, 0.0],
                    "_score": 0.95
                },
                ...
            ]
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for hybrid search.")

        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {metric_result}")
        
        if not connections.has_connection(alias="default"):
            # If no connection exists, create a new one
            connections.connect(uri=self.config.uri, token=self.config.token, _async=self.run_async)

        # Construct the collection
        self.collection = Collection(collection_name)

        try:
            search_requests = self.build_hybrid_search_requests(
                embedding_data=embedding_data,
                top_k=top_k,
                metric_type=metric_type,
                index_type=index_type
            )
            if not search_requests:
                raise SearchMilvusVectorsError("No valid search requests were created. Check the input embedding data.")
            
            results = await self.async_client.hybrid_search(
                collection_name=collection_name,
                reqs=search_requests,
                ranker=self.reranker,
                limit=top_k,
                output_fields=output_fields,
                **kwargs
            )

            if not results:
                return []

            return results[0]
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in hybrid search: {str(e)}")
    
    def search_dense_vectors(
        self,
        query_embeddings: List[List],
        field_name: str,
        output_fields: List[str],
        filtering_expr: str = "",
        top_k: int = 5,
        metric_type: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[dict]]:
        """
        Search for dense vectors in the collection in Milvus database.
        
        Args:
            collection_name (str): Name of the collection.
            query_embeddings (List[List]): List of query embeddings.
            field_name (str): Field name to search.
            output_fields (List[str]): List of fields to return in the search results.
            filtering_expr (str): Filtering expression for the search query.
            top_k (int): Number of results to return.
            metric_type (MetricType): Metric type for the search query.
            index_type (IndexType): Index type for dense vector ("FLAT", "IVF_FLAT", "HNSW").

        Returns:
            List[List[dict]]: List of top-k search results of each input query embedding. 
                The number of lists in the output is equal to the number of query embeddings.
                The output contains the expected output fields and "_score" key.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for searching dense vectors.")

        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {metric_result}")
        
        if not connections.has_connection(alias="default"):
            # If no connection exists, create a new one
            connections.connect(uri=self.config.uri, token=self.config.token, _async=self.run_async)

        # Construct the collection
        self.collection = Collection(collection_name)

        try:
            results = self.client.search(
                collection_name=collection_name,
                data=query_embeddings,
                anns_field=field_name,
                limit=top_k,
                output_fields=output_fields,
                search_params={
                    "metric_type": metric_type.value,
                    "params": {"ef": top_k} if index_type == IndexType.HNSW else {"nprobe": 8},
                },
                filter=filtering_expr,
                **kwargs
            )
                
            # Flatten the structure for each query result
            flattened_results = []
            for query_result in results:
                flattened_query_result = []
                for result in query_result:
                    flattened_result = {}
                    
                    # Copy non-entity fields (like distance, id, etc.)
                    for key, value in result.items():
                        if key != "entity":
                            if key == "distance":
                                flattened_result["_score"] = value
                            else:
                                flattened_result[key] = value
                    
                    # Move entity fields to top level
                    if "entity" in result:
                        for entity_key, entity_value in result["entity"].items():
                            flattened_result[entity_key] = entity_value
                    
                    flattened_query_result.append(flattened_result)
                
                flattened_results.append(flattened_query_result)
                
            return flattened_results
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {str(e)}")
        
    async def async_search_dense_vectors(
        self,
        query_embeddings: List[List],
        field_name: str,
        output_fields: List[str],
        filtering_expr: str = "",
        top_k: int = 5,
        metric_type: MetricType = MetricType.COSINE,
        index_type: IndexType = IndexType.HNSW,
        collection_name: str = None,
        **kwargs,
    ) -> List[List[dict]]:
        """
        Asynchronously search for dense vectors in the collection in Milvus database.
        
        Args:
            collection_name (str): Name of the collection.
            query_embeddings (List[List]): List of query embeddings.
            field_name (str): Field name to search.
            output_fields (List[str]): List of fields to return in the search results.
            filtering_expr (str): Filtering expression for the search query.
            top_k (int): Number of results to return.
            metric_type (MetricType): Metric type for the search query.
            index_type (IndexType): Index type for dense vector ("FLAT", "IVF_FLAT", "HNSW").

        Returns:
            List[List[dict]]: List of top-k search results of each input query embedding. 
                The number of lists in the output is equal to the number of query embeddings.
                The output contains the expected output fields and "_score" key.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided for searching dense vectors.")

        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {metric_result}")
        
        if not connections.has_connection(alias="default"):
            # If no connection exists, create a new one
            connections.connect(uri=self.config.uri, token=self.config.token, _async=self.run_async)

        # Construct the collection
        self.collection = Collection(collection_name)

        try:
            results = await self.async_client.search(
                collection_name=collection_name,
                data=query_embeddings,
                anns_field=field_name,
                limit=top_k,
                output_fields=output_fields,
                search_params={
                    "metric_type": metric_type.value,
                    "params": {"ef": top_k} if index_type == IndexType.HNSW else {"nprobe": 8},
                },
                filter=filtering_expr,
                **kwargs
            )

            # Flatten the structure for each query result
            flattened_results = []
            for query_result in results:
                flattened_query_result = []
                for result in query_result:
                    flattened_result = {}
                    
                    # Copy non-entity fields (like distance, id, etc.)
                    for key, value in result.items():
                        if key != "entity":
                            if key == "distance":
                                flattened_result["_score"] = value
                            else:
                                flattened_result[key] = value
                    
                    # Move entity fields to top level
                    if "entity" in result:
                        for entity_key, entity_value in result["entity"].items():
                            flattened_result[entity_key] = entity_value
                    
                    flattened_query_result.append(flattened_result)
                
                flattened_results.append(flattened_query_result)
                
            return flattened_results
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {str(e)}")
        