import numpy as np
import supervision as sv
import torch
from PIL import Image

from conceptgraph.utils.general_utils import filter_detections
from conceptgraph.utils.model_utils import (
    _make_mask_focused_crop,
    compute_clip_features_batched,
)


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


def test_make_mask_focused_crop_preserves_foreground_and_suppresses_context():
    image = np.full((7, 7, 3), 200, dtype=np.uint8)
    image[3, 3] = [20, 40, 60]
    mask = np.zeros((7, 7), dtype=np.bool_)
    mask[3, 3] = True

    focused = np.asarray(
        _make_mask_focused_crop(
            Image.fromarray(image),
            mask,
            (0, 0, 7, 7),
            background_factor=0.1,
            blur_radius=0.0,
        )
    )

    np.testing.assert_array_equal(focused[3, 3], image[3, 3])
    np.testing.assert_array_equal(focused[0, 0], [20, 20, 20])


class _TwoViewClipModel:
    visual = _DummyVisualEncoder()

    def __init__(self):
        self.calls = 0

    def encode_image(self, batch):
        self.calls += 1
        feature = [1.0, 0.0] if self.calls == 1 else [0.0, 1.0]
        return torch.tensor([feature] * len(batch), dtype=torch.float32)


def test_compute_clip_features_batched_fuses_normalized_views_equally():
    detections = sv.Detections(
        xyxy=np.array([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32),
        mask=np.ones((1, 6, 6), dtype=np.bool_),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int64),
    )
    model = _TwoViewClipModel()

    _, image_feats, _ = compute_clip_features_batched(
        image=np.zeros((6, 6, 3), dtype=np.uint8),
        detections=detections,
        clip_model=model,
        clip_preprocess=lambda image: torch.zeros((3, 2, 2)),
        clip_tokenizer=lambda labels: torch.zeros((len(labels), 1)),
        classes=np.array(["object"]),
        device="cpu",
        bbox_padding=0,
        masked_weight=0.5,
    )

    assert model.calls == 2
    np.testing.assert_allclose(
        image_feats,
        [[2 ** -0.5, 2 ** -0.5]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_clip_features_batched_can_restore_bbox_only_without_masks():
    detections = sv.Detections(
        xyxy=np.array([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int64),
    )
    model = _TwoViewClipModel()

    _, image_feats, _ = compute_clip_features_batched(
        image=np.zeros((6, 6, 3), dtype=np.uint8),
        detections=detections,
        clip_model=model,
        clip_preprocess=lambda image: torch.zeros((3, 2, 2)),
        clip_tokenizer=lambda labels: torch.zeros((len(labels), 1)),
        classes=np.array(["object"]),
        device="cpu",
        bbox_padding=0,
        masked_weight=0.0,
    )

    assert model.calls == 1
    np.testing.assert_allclose(image_feats, [[1.0, 0.0]])
