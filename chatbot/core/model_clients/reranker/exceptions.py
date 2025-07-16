class RerankerError(Exception):
    """Base exception class for all reranker-related errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CallServerRerankerError(RerankerError):
    """Exception raised when there's an error calling the reranker server."""
    pass


class RerankerConfigurationError(RerankerError):
    """Exception raised when there's a configuration error in the reranker."""
    pass


class RerankerTimeoutError(RerankerError):
    """Exception raised when a reranker operation times out."""
    pass
