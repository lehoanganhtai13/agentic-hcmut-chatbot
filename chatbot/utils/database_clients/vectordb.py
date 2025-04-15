from typing import Dict, List, Optional

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
    RRFRanker,
)

from chatbot.utils.database_clients.base_class import EmbeddingData, IndexParam
from chatbot.utils.database_clients.exceptions import (
    CreateMilvusCollectionError,
    InsertMilvusVectorsError,
    GetMilvusVectorsError,
    SearchMilvusVectorsError
)


class VectorDatabase:
    """MilvusClient class for vector database operations."""
    def __init__(self, host: str = "localhost", port: str = "19530", run_async: bool = False) -> None:
        self.uri = f"http://{host}:{port}"
        self.client = AsyncMilvusClient(uri=f"http://{host}:{port}") if run_async else MilvusClient(uri=f"http://{host}:{port}")
        self.reranker = RRFRanker()
        self.run_async = run_async

    def create_collection(
        self,
        collection_name: str,
        collection_structure: List[Dict],
        auto_id: bool = False,
        enable_dynamic_field: bool = False,
        json_index_params: Dict[str, List[IndexParam]] = None
    ):
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): Name of the collection to create.
            collection_structure (List[Dict]): List of dictionaries containing the field structure.
            auto_id (bool): Enable auto ID generation for the collection.
            enable_dynamic_field (bool): Enable dynamic field for the collection allowing new fields to be added.
            json_index_params (Dict[str, List[IndexParam]]): Index parameters of JSON type for the collection.
                Key is the field name and value is a list of IndexParam objects.
        """
        # Check if collection exists
        if self.client.has_collection(collection_name):
            if self.run_async:
                asyncio.run(self.client.drop_collection(collection_name))
            else:
                self.client.drop_collection(collection_name)

        data_type_mapping = {
            "int": DataType.INT64,
            "float": DataType.FLOAT_VECTOR,
            "sparse_float": DataType.SPARSE_FLOAT_VECTOR,
            "string": DataType.VARCHAR,
            "array": DataType.ARRAY,
            "bool": DataType.BOOL,
            "json": DataType.JSON
        }

        fields = []
        index_params = self.client.prepare_index_params()

        for field in collection_structure:
            if field["field_type"] == "int":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    is_primary=field.get("is_primary", False)
                )
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type="STL_SORT"
                )
            elif field["field_type"] == "float":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    dim=field["dimension"],
                    is_primary=field.get("is_primary", False)
                )

                index_result = self.check_index_type(field["index_type"])
                if index_result != field["index_type"]:
                    raise CreateMilvusCollectionError(f"Error creating collection: {index_result}")
                
                metric_result = self.check_metric_type(field["metric_type"])
                if metric_result != field["metric_type"]:
                    raise CreateMilvusCollectionError(f"Error creating collection: {metric_result}")
                
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type=field["index_type"],
                    metric_type=field["metric_type"],
                    params=field["index_params"]
                )
            elif field["field_type"] == "sparse_float":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    is_primary=field.get("is_primary", False)
                )
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP"
                )
            elif field["field_type"] == "string":
                analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    max_length=65535,
                    analyzer_params=analyzer_params,
                    enable_analyzer=True,
                    enable_match=True,
                    is_primary=field.get("is_primary", False)
                )
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type="INVERTED"
                )
            elif field["field_type"] == "array":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    element_type=DataType.VARCHAR if field.get("element_type", "string") == "string" else DataType.INT64,
                    max_capacity=field.get("max_capacity", 100),
                    max_length=65535,
                    is_primary=field.get("is_primary", False)
                )
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type="AUTOINDEX"
                )
            elif field["field_type"] == "bool":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    is_primary=field.get("is_primary", False)
                )
                index_params.add_index(
                    field_name=field["field_name"],
                    index_type="AUTOINDEX"
                )
            elif field["field_type"] == "json":
                schema_field = FieldSchema(
                    name=field["field_name"],
                    dtype=data_type_mapping[field["field_type"]],
                    description=field["field_description"],
                    is_primary=field.get("is_primary", False)
                )

                # Add index parameters for JSON fields
                if json_index_params and json_index_params.get(field["field_name"], None):
                    for idx, index_param in enumerate(json_index_params[field["field_name"]]):
                        index_params.add_index(
                            field_name=field["field_name"],
                            index_type="INVERTED",
                            index_name=index_param.index_name,
                            params={
                                "json_path": index_param.indexed_key,
                                "json_cast_type": index_param.value_type.value
                            }
                        )
                            
            else:
                raise ValueError("Invalid field type. Please provide one of 'int', 'float', 'sparse_float', 'string'")
            
            fields.append(schema_field)
        
        schema = CollectionSchema(
            fields=fields,
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field
        )

        # Create collection
        if self.run_async:
            asyncio.run(
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params
                )
            )
        else:
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

        logger.info(f"Collection {collection_name} created successfully!")

    def check_metric_type(self, metric_type: str) -> str:
        """
        Check if the metric type is supported.
        
        Args:
            metric_type (str): Metric type of the index.

        Returns:
            str: Metric type if supported, error message otherwise.
        """
        supported_metric_types = ["L2", "COSINE"]
        if metric_type not in supported_metric_types:
            return f"Invalid metric type. Please provide one of {supported_metric_types}"
        return metric_type
    
    def check_index_type(self, index_type: str) -> str:
        """
        Check if the index type is supported.
        
        Args:
            index_type (str): Index type of the index.

        Returns:
            str: Index type if supported, error message otherwise.
        """
        supported_index_types = ["IVF_FLAT", "HNSW"]
        if index_type not in supported_index_types:
            return f"Invalid index type. Please provide one of {supported_index_types}"
        return index_type

    def load_collection(self, collection_name: str) -> bool:
        """
        Load the collection into memory for faster search operations.

        Args:
            collection_name (str): Name of the collection to load.

        Returns:
            bool: True if the collection is loaded successfully, False otherwise.
        """
        if not self.client.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist!")
            return
        
        # Load the collection
        if self.run_async:
            asyncio.run(self.client.load_collection(collection_name))
        else:
            self.client.load_collection(collection_name)

        # Check if the collection is loaded
        load_state = self.client.get_load_state(collection_name=collection_name)
        if load_state:
            logger.info(f"Collection {collection_name} is loaded successfully!")
            return True
        else:
            logger.warning(f"Failed to load collection {collection_name}!")
            return False

    def insert_vectors(
        self,
        collection_name: str,
        data: Dict
    ) -> None:
        try:
            if self.run_async:
                asyncio.run(
                    self.client.insert(
                        collection_name=collection_name,
                        data=data
                    )
                )
            else:
                self.client.insert(
                    collection_name=collection_name,
                    data=data
                )
        except Exception as e:
            raise InsertMilvusVectorsError(f"Error inserting vectors: {str(e)}")
    
    def get_vectors(self, collection_name: str, ids: List[str]) -> List[dict]:
        """
        Get vectors from the collection by their IDs.

        Args:
            collection_name (str): Name of the collection.
            ids (List[int]): List of IDs to retrieve.

        Returns:
            List[dict]: List of dictionaries containing the vectors and metadata.
        """
        try:
            result = []
            if self.run_async:
                result = asyncio.run(
                    self.client.get(
                        collection_name=collection_name,
                        ids=ids
                    )
                )
            else:
                result = self.client.get(
                    collection_name=collection_name,
                    ids=ids
                )
            return result
        except Exception as e:
            raise GetMilvusVectorsError(f"Error getting vectors: {str(e)}")

    def hybrid_search_vectors(
        self,
        collection_name: str,
        dense_data: EmbeddingData,
        sparse_data: EmbeddingData,
        output_fields: List[str],
        top_k: int = 5,
        metric_type: str = "COSINE",
        index_type: str = "HNSW"
    ) -> List[List[dict]]:
        """
        Perform hybrid search (dense + sparse) for vectors in the collection.

        Args:
            collection_name (str): Name of the collection.
            dense_data (EmbeddingData): Dense data for the search queries
            sparse_data (EmbeddingData): Sparse data for the search queries.
            output_fields (List[str]): List of fields to return in the search results.
            top_k (int): Number of results to return.
            metric_type (str): Metric type for the search query.
            index_type (str): Index type for the search query.

        Returns:
            List[List[dict]]: List of hybrid search results (based on number of input queries).
                Each list contains the top-k search results for each input query.
                Each result is a dictionary containing the expected output fields.
        """
        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in hybrid search: {metric_result}")
        
        if not connections.has_connection(alias="default"):
            # If no connection exists, create a new one
            connections.connect(uri=self.uri, _async=self.run_async)

        # Construct the collection
        self.collection = Collection(collection_name)

        try:
            search_requests = []
            for i in range(len(dense_data.embeddings)):
                # Create dense search request
                dense_search_params = {
                    "data": [dense_data.embeddings[i]],
                    "anns_field": dense_data.field_name,
                    "param": {
                        "metric_type": metric_type,
                        "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8}
                    },
                    "limit": top_k,
                    "expr": dense_data.filtering_expr
                }
                dense_search_request = AnnSearchRequest(**dense_search_params)

                # Create sparse search request
                sparse_search_params = {
                    "data": [sparse_data.embeddings[i]],
                    "anns_field": sparse_data.field_name,
                    "param": {
                        "metric_type": "IP",
                        "params": {}
                    },
                    "limit": top_k,
                    "expr": sparse_data.filtering_expr
                }
                sparse_search_request = AnnSearchRequest(**sparse_search_params)

                search_requests.extend([sparse_search_request, dense_search_request])
            
            results = []
            if self.run_async:
                results = asyncio.run(
                    self.client.hybrid_search(
                        collection_name=collection_name,
                        reqs=search_requests,
                        ranker=self.reranker,
                        limit=top_k,
                        output_fields=output_fields
                    )
                )
            else:
                results = self.client.hybrid_search(
                    collection_name=collection_name,
                    reqs=search_requests,
                    ranker=self.reranker,
                    limit=top_k,
                    output_fields=output_fields
                )
            return results
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in hybrid search: {str(e)}")

    def search_desnse_vectors(
        self,
        collection_name: str,
        query_embeddings: List[List],
        field_name: str,
        output_fields: List[str],
        filtering_expr: Optional[str] = "",
        top_k: int = 5,
        metric_type: str = "COSINE",
        index_type: str = "HNSW"
    ) -> List[List[dict]]:
        """
        Search for dense vectors in the collection in Milvus database.
        
        Args:
            collection_name (str): Name of the collection.
            query_embeddings (List[List]): List of query embeddings.
            field_name (str): Field name to search.
            output_fields (List[str]): List of fields to return in the search results.
            filtering_expr (Optional[str]): Filtering expression for the search query.
            top_k (int): Number of results to return.
            metric_type (str): Metric type for the search query.
            index_type (str): Index type for the search query.

        Returns:
            List[List[dict]]: List of top-k search results of each input query embedding. 
                The number of lists in the output is equal to the number of query embeddings.
        """
        index_result = self.check_index_type(index_type)
        if index_result != index_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {index_result}")
        
        metric_result = self.check_metric_type(metric_type)
        if metric_result != metric_type:
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {metric_result}")

        try:
            results = []
            if self.run_async:
                results = asyncio.run(
                    self.client.search(
                        collection_name=collection_name,
                        data=query_embeddings,
                        anns_field=field_name,
                        limit=top_k,
                        output_fields=output_fields,
                        search_params={
                            "metric_type": metric_type,
                            "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8},
                        },
                        filter=filtering_expr
                    )
                )
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    data=query_embeddings,
                    anns_field=field_name,
                    limit=top_k,
                    output_fields=output_fields,
                    search_params={
                        "metric_type": metric_type,
                        "params": {"ef": top_k} if index_type == "HNSW" else {"nprobe": 8},
                    },
                    filter=filtering_expr
                )
            return results
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchMilvusVectorsError(f"Error in searching dense vectors: {str(e)}")
        