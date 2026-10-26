import json

import make_test_vectors


def test_shared_vectors_are_up_to_date():
    """Fails if chessnet/encode.py changed without re-running make_test_vectors.py."""
    assert make_test_vectors.OUT.exists(), "run: python make_test_vectors.py"
    assert json.loads(make_test_vectors.OUT.read_text()) == make_test_vectors.build()
