class ConnectLanceDBDatabaseError(Exception):
    """Exception raised for errors in connecting to the LanceDB database."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InsertLanceDBVectorError(Exception):
    """Exception raised for errors in inserting data into the LanceDB database."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
