
import numpy as np
from scipy.cluster.vq import kmeans2, vq


class PQ(object):
    """Python implementation of Product Quantization (PQ).
    For the indexing phase of database vectors,  a `d`-length input vector is divided into `m` `d`/`m`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `k-means` with `k` centroids.

    For querying, given a new `d`-dim query vector, the distance between query and each database PQ-codes is
    efficiently approximated via Asymmetric Distance Computation (ADC).
    All vectors must be np.ndarray with dtype np.float32
    ref: H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
    Attributes:
        m (int): The number of sub-space
        k (int): The number of codewords for each subspace
        verbose (bool): Verbose flag
        code_dtype (object): dtype of PQ-code. Either np.uint{8, 16, 32}
        code_books (np.ndarray): shape=(m, k, s) with dtype=np.float32.
            code_books[m][k] means k-th codeword (s-dim) for m-th subspace
        s (int): The dim of each sub-vector, i.e., s = d/m
    """
    def __init__(self, m: int, k: int = 256, verbose: bool = True):
        """
        :param m: The number of sub-vectors for original `d` length vector.
        :param k: The number of centroids for each group of sub-vectors, typically 256,
                  so that each sub-vector is quantized into 256 bits = 1 byte = uint8
        :param verbose: Verbose flag.
        """
        if not 0 <= k <= 2 ** 32:
            raise ValueError("The number of codewords should be between (0, 2 ** 32)")
        if k <= 2 ** 8:  # (256)
            self.code_dtype = np.uint8
        elif k <= 2 ** 16:
            self.code_dtype = np.uint16
        else:
            self.code_dtype = np.uint32
        self.m = m
        self.k = k
        self.verbose = verbose
        self.code_books = None
        self.s = None

        if self.verbose:
            print(f"M: {self.m}, K: {self.k}, code_dtype: {self.code_dtype}")

    def __repr__(self):
        return f"PQ(m={self.m}, k={self.k}, verbose={self.verbose})"

    @staticmethod
    def _check_input_vectors(vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("The number of dims of vectors should be 2.")
        return vectors.astype("float32")

    def fit(self, vectors: np.ndarray, epochs: int = 20, random_state: int = 123):
        """Given training vectors, run k-means for each sub-space and create codewords for each sub-space.
        This function should be run once first of all.
        :param vectors: Training vectors with shape=(N, D) and dtype=np.float32.
        :param epochs: The number of iteration for k-means
        :param random_state: The seed for random process
        """
        vectors = self._check_input_vectors(vectors)
        n, d = vectors.shape
        if n <= self.k:
            raise ValueError(f"The number of training vectors should be more than k(e.g. {self.k})")
        if d % self.m != 0:
            raise ValueError(f"The length of each vector should be dividable by m(e.g. {self.m})")
        self.s = d // self.m

        np.random.seed(random_state)
        self.code_books = np.zeros(shape=(self.m, self.k, self.s), dtype=np.float32)

        for i in range(self.m):
            if self.verbose:
                print(f"Training for {i + 1} / {self.m} subspace.")
            sub_vectors = vectors[:, i * self.s: (i + 1) * self.s]
            centroids, _ = kmeans2(sub_vectors, k=self.k, iter=epochs, minit="points")
            self.code_books[i] = centroids
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode given vectors into PQ-codes
        :param vectors: with shape=(*, d)
        :returns: PQ-codes array with shape=(*, m) and dtype=self.code_dtype
        """
        if self.code_books is None:
            raise RuntimeError("The PQ object has not been trained, please run `fit` with train vectors.")
        vectors = self._check_input_vectors(vectors)
        n, d = vectors.shape
        if d != self.m * self.s:
            raise ValueError(f"The input vector's length must be same with train vector (e.g. {self.m * self.s})")

        codes = np.empty(shape=(n, self.m), dtype=self.code_dtype)
        for i in range(self.m):
            if self.verbose:
                print(f"Encode for {i + 1} / {self.m} subspace.")
            sub_vectors = vectors[:, i * self.s: (i + 1) * self.s]
            codes[:, i], _ = vq(sub_vectors, code_book=self.code_books[i])
        return codes

    def decode(self, codes: np.ndarray):
        """
        Given PQ-codes, reconstruct original D-dimensional vectors approximately by fetching the codewords.
        :param codes: PQ-codes with shape=(n, m) and dtype=self.code_dtype. Each row is a PQ-code
        :return: np.ndarray: Reconstructed vectors with shape=(n, d) and dtype=np.float32
        """
        if codes.ndim != 2:
            raise ValueError("The number of dims of codes should be 2.")
        n, m = codes.shape
        if m != self.m:
            raise ValueError(f"The number of sub-space should be same as PQ's e.g. {self.m}")

        vectors = np.empty(shape=(n, self.m * self.s), dtype=np.float32)
        for i in range(self.m):
            sub_codes = codes[:, i]
            vectors[:, i * self.s: (i + 1) * self.s] = self.code_books[i][sub_codes, :]
        return vectors

    def distance_table(self, query: np.ndarray):
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query to the codewords for each sub-subspace.
        `table[m][k]` contains the squared Euclidean distance between the `m`-th sub-vector of the query and
        the `k`-th codeword
        :param query: Input vector with shape=(D, ) and dtype=np.float32
        :returns: Distance table.with shape=(M, Ks) and dtype=np.float32, which holds the squared Euclidean distance
        """
        if query.ndim != 1:
            raise ValueError("The input should be a single vector and flatten.")
        d = len(query)
        if d != self.s * self.m:
            raise ValueError(f"The length of query vector should be sam eas PQ's e.g. {self.s * self.m}")

        query = query.astype("float32")
        table = np.empty(shape=(self.m, self.k), dtype=np.float32)

        for i in range(self.m):
            query_sub = query[i * self.s: (i + 1) * self.s]
            table[i, :] = np.linalg.norm(self.code_books[i] - query_sub, axis=1) ** 2  # The squared distance


class DistanceTable(object):
    """
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.dist`.
    Attributes:
        table: Squared-distance table with shape (m, k)
        m, k: The shape of table.
    """
    def __init__(self, table: np.ndarray):
        """

        """
        if table.ndim != 2:
            raise ValueError("The number of dim of table should be 2.")
        self.table = table
        self.m, self.k = self.table.shape

    def dist(self, codes: np.ndarray):
        """Given PQ-codes, compute Asymmetric Distances between query and each PQ-code"""
        if codes.ndim != 2:
            raise ValueError("The number of dim of codes should be 2.")
        n, m = codes.shape
        if m != self.m:
            raise ValueError(f"The number of sub-space should be same as table's e.g. {self.m}")

        # Complex indexing for numpy, the `range(m)` will be broadcast as (n, m) as same shape as codes
        # and the result returned is based on zipped coordinates
        dists = self.table[range(m), codes]   # [n, m]
        dists = np.sum(dists, axis=1)

        # The above is equivalent to the following:
        # dists = np.zeros(shape=(n, ), dtype=np.float32)
        # for i in range(n):
        #     for j in range(m):
        #         dists[i] += self.table[i, codes[i, j]]

        return dists
