class CallServerEmbedderError(Exception):
    """Exception raised for errors in the call to the Embedder server."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
