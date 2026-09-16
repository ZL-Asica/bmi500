"""Reference implementations for vector and matrix-vector multiplication."""

from __future__ import annotations

import random
import math
import time
from collections.abc import Sequence
from numbers import Real


def dot_product(vector_a: Sequence[Real], vector_b: Sequence[Real]) -> Real:
    """Sum pairwise products with a for-loop; empty vectors have product zero.

    Inputs must contain finite real numbers, excluding booleans. Incompatible
    lengths raise ValueError and nonnumeric entries raise TypeError.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")
    total = 0
    for a, b in zip(vector_a, vector_b):
        for value in (a, b):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("vector entries must be real numbers, excluding bool")
            if not isinstance(value, int) and not math.isfinite(value):
                raise ValueError("vector entries must be finite")
        total += a * b
    return total



def matvec_multiply(matrix: Sequence[Sequence[Real]], vector: Sequence[Real]) -> list[Real]:
    """Return ``matrix @ vector`` (one result per row) using dot_product.

    All rows must have the vector's length. A zero-row matrix returns []; an
    m-by-0 matrix multiplied by [] returns m zeros. Inputs are not modified.
    """
    # Validate the vector even when there are no rows, so invalid inputs never
    # pass silently merely because the result happens to be empty.
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("vector entries must be real numbers, excluding bool")
        if not isinstance(value, int) and not math.isfinite(value):
            raise ValueError("vector entries must be finite")
    if len(matrix) == 0:
        return []
    width = len(vector)
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must all match the vector length")
    return [dot_product(row, vector) for row in matrix]



def main() -> None:
    """Demonstrate a reproducible 1000-by-1000 product and verify every row."""
    size = 1_000
    rng = random.Random(500)
    matrix = [[rng.random() for _ in range(size)] for _ in range(size)]
    vector = [rng.random() for _ in range(size)]
    started = time.perf_counter()
    result = matvec_multiply(matrix, vector)
    elapsed = time.perf_counter() - started
    reference = [math.fsum(a * b for a, b in zip(row, vector)) for row in matrix]
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(result, reference))
    print(f"Verified {size}x{size} matrix-vector product in {elapsed:.6f} seconds.")
    print(f"First three results: {result[:3]}")


if __name__ == "__main__":
    main()
