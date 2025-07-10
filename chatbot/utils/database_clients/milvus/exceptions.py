class CreateMilvusCollectionError(Exception):
    """Exception raised for errors in creating a Milvus collection."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InsertMilvusVectorsError(Exception):
    """Exception raised for errors in inserting vector in Milvus collection."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GetMilvusItemsError(Exception):
    """Exception raised for errors in getting items in Milvus collection."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SearchMilvusVectorsError(Exception):
    """Exception raised for errors in searching vectors in Milvus collection."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
