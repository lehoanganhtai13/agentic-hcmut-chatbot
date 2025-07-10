from chatbot.utils.database_clients.base_class import (
    VectorDBBackend,
    VectorDBConfig
)


class MilvusConfig(VectorDBConfig):
    """
    Configuration for Milvus database.

    Args:
        host (str): Hostname of the Milvus server. Defaults to "localhost".
        port (str): Port number of the Milvus server. Defaults to "19530".
        cloud_uri (str, optional): Cloud URI for Milvus. If provided, overrides host and port.
        token (str, optional): Authentication token for Milvus cloud connection.
        run_async (bool): Whether to run the client asynchronously. Defaults to False.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        cloud_uri: str = "",
        token: str = "",
        run_async: bool = False,
        **kwargs
    ):
        super().__init__(VectorDBBackend.MILVUS, **kwargs)
        self.host = host
        self.port = port
        self.run_async = run_async
        self.uri = cloud_uri or f"http://{host}:{port}"
        self.token = token
