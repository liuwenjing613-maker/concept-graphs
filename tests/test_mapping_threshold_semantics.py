import torch

from conceptgraph.slam.mapping import (
    SIMILARITY_THRESHOLD_COMPARATOR,
    match_detections_to_objects,
    similarity_exceeds_threshold,
)


def test_similarity_threshold_is_strict_greater_than():
    assert SIMILARITY_THRESHOLD_COMPARATOR == "STRICT_GREATER_THAN"
    assert not similarity_exceeds_threshold(1.2, 1.2)
    assert not similarity_exceeds_threshold(1.1999999, 1.2)
    assert similarity_exceeds_threshold(1.2000001, 1.2)


def test_native_matcher_creates_at_exact_threshold():
    scores = torch.tensor(
        [
            [1.2, 1.1],
            [1.2000001, 0.4],
            [1.1999999, 1.0],
        ],
        dtype=torch.float64,
    )

    assert match_detections_to_objects(scores, 1.2) == [None, 0, None]
