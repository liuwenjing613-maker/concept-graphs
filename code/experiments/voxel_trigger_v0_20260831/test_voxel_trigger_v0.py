#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import voxel_trigger_v0 as module


def main() -> None:
    observations = [
        module.pack_coords(np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)),
        module.pack_coords(np.asarray([[0, 0, 0], [5, 0, 0]], dtype=np.int32)),
        module.pack_coords(np.asarray([[1, 0, 0]], dtype=np.int32)),
    ]
    labels = np.asarray([2, 3, 3], dtype=np.int32)
    with tempfile.TemporaryDirectory() as directory:
        result = module.build_voxel_map(
            observation_voxels=observations,
            observation_labels=labels,
            selected=np.asarray([True, True, True]),
            output=Path(directory) / "voxel_map.npz",
        )
        assert result.seen_count.tolist() == [2, 2, 1]
        object_keys = module.pack_coords(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
        )
        evidence, histogram = module.object_evidence(object_keys, result)
        assert histogram == {2: 2, 3: 2}
        assert evidence["evidence_supported_voxels"] == 2
        assert evidence["stable_evidence_voxels"] == 2
        assert evidence["primary_evidence_label_id"] == 2
        assert evidence["mean_seen_count"] == 2.0
        saved = np.load(Path(directory) / "voxel_map.npz")
        assert set(saved.files) == {
            "schema_version",
            "voxel_keys",
            "voxel_coords",
            "seen_count",
            "obs_offsets",
            "obs_ids",
            "label_offsets",
            "label_ids",
            "label_counts",
        }
    print("voxel payload + object derivation: PASS")


if __name__ == "__main__":
    main()
