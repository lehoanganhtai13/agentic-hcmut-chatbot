from typing import List, Union

import numpy as np
from scipy.sparse import csr_array


# ------------------- Embedding Cache Classes -------------------


DenseEmbedding = List[float]                            # Type alias for embedding vectors
BinaryEmbedding = Union[List[int], bytes, np.ndarray]    # Type alias for binary embeddings (0s and 1s)
SparseEmbedding = csr_array 
