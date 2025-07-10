from enum import Enum

from chatbot.utils.database_clients.base_class import (
    VectorDBBackend,
    VectorDBConfig
)


class HostType(Enum):
    """
    Enum for different types of hosts for LanceDB.
    
    Attributes:
        LOCAL: Local host type.
        CLOUD: Cloud host type.
        MINIO: MinIO host type.
    """
    LOCAL = "local"
    CLOUD = "cloud"
    MINIO = "minio"


class LanceDBConfig(VectorDBConfig):
    """
    Configuration for LanceDB database.
    
    Attributes:
        local_path (str, optional): Local path for LanceDB storage.
        cloud_uri (str, optional): URI for cloud storage of LanceDB data.
        host (str): The host of the MinIO server for storing LanceDB data.
        port (str): The port of the MinIO server for storing LanceDB data.
        api_key (str, optional): API key for cloud access.
        region (str, optional): Region for cloud storage.
        bucket_name (str, optional): Name of the bucket for storing database data.
        prefix (str, optional): Prefix for the bucket path.
        aws_access_key_id (str, optional): AWS access key ID for authentication.
        aws_secret_access_key (str, optional): AWS secret access key for authentication.
        secure (bool): Whether to use a secure connection (HTTPS).
        host_type (HostType): Type of host for LanceDB ('local', 'cloud', 'minio').
        **kwargs: Additional keyword arguments for configuration.
    """
    def __init__(
        self,
        local_path: str = None,
        cloud_uri: str = None,
        host: str = "localhost",
        port: str = "9000",
        api_key: str = None,
        region: str = None,
        bucket_name: str = None,
        prefix: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        secure: bool = False,
        host_type: HostType = HostType.LOCAL,
        **kwargs
    ):
        super().__init__(VectorDBBackend.LANCEDB, **kwargs)
        self.local_path = local_path
        self.cloud_uri = cloud_uri
        self.api_key = api_key
        self.region = region
        self.endpoint = f"http://{host}:{port}"
        self.bucket_name = bucket_name if bucket_name else ""
        self.prefix = prefix if prefix else ""
        self.bucket_path = f"s3://{bucket_name}/{prefix}" if bucket_name else f"s3://{bucket_name}"
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.secure = secure
        self.host_type = host_type
