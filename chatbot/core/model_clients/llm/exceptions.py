class CallServerLLMError(Exception):
    """Exception raised for errors in the call to the LLM server."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        