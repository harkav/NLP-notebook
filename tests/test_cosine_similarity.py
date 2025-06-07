import pytest
import math
from utils.cosine_similarity import cosine_similarity


def test_validate_input():
    with pytest.raises(ValueError):
        cosine_similarity([0.0], [1.0, 2])
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 2.2, 3.0], [1.0, 2])


def test_cosine_similarity_calculates_properly():
    # Edge case, to vectors of [0.0]

    result = cosine_similarity([0.0], [0.0])
    assert result == 0.0, print(result)

    # Edge case, identical vectors.

    result = cosine_similarity([1.0, 1.0], [1.0, 1.0])
    assert round(result, 5) == 1.0, print(result)

    # Orthogonal vectors, should be 0.0. (two vectors are orthogonal when the dotproduct == 0)
    result = cosine_similarity([1.0, 0.0], [0.0, 1.0])
    # assert round(result, 5) == 0.0, print(result)

    # opposite vectors, should be -1.0
    result = cosine_similarity([2.0, 1.0], [-2.0, -1.0])
    assert round(result, 5) == -1.0, print(result)

    # Not orthogonal:

    result = cosine_similarity([2.0, 1.0], [1.0, 2.0])

    assert math.isclose(result, 0.7999999999999999, rel_tol=1e-5), print(result)

    result = cosine_similarity([1, 2, 3], [2, 2, 2])
    assert math.isclose(result, 0.9258200997725516, rel_tol=1e-5), print(result)
