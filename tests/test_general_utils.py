import numpy as np
import supervision as sv

from conceptgraph.utils.general_utils import filter_detections
from conceptgraph.utils.model_utils import compute_clip_features_batched


class _SingleClassCatalog:
    bg_classes = []

    @staticmethod
    def get_classes_arr():
        return np.array(["tiny-object"])


def test_filter_detections_returns_well_formed_empty_batch():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        mask=np.zeros((1, 10, 12), dtype=np.bool_),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int64),
    )

    filtered, labels = filter_detections(
        image=image,
        detections=detections,
        classes=_SingleClassCatalog(),
        given_labels=["tiny-object 0"],
    )

    assert labels == []
    assert len(filtered) == 0
    assert filtered.xyxy.shape == (0, 4)
    assert filtered.mask.shape == (0, 10, 12)
    assert filtered.confidence.shape == (0,)
    assert filtered.class_id.shape == (0,)


class _DummyVisualEncoder:
    output_dim = 7


class _DummyClipModel:
    visual = _DummyVisualEncoder()


def _must_not_run(*args, **kwargs):
    raise AssertionError("empty detection batches must bypass CLIP preprocessing")


def test_compute_clip_features_batched_accepts_empty_detections():
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=np.empty((0, 10, 12), dtype=np.bool_),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
    )

    crops, image_feats, text_feats = compute_clip_features_batched(
        image=np.zeros((10, 12, 3), dtype=np.uint8),
        detections=detections,
        clip_model=_DummyClipModel(),
        clip_preprocess=_must_not_run,
        clip_tokenizer=_must_not_run,
        classes=np.array(["unused"]),
        device="cpu",
    )

    assert crops == []
    assert image_feats.shape == (0, 7)
    assert image_feats.dtype == np.float32
    assert text_feats == []
