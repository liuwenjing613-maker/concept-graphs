import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from evaluation.observation_gt.build_replica_official_reference import build_reference
from evaluation.observation_gt.observation_labeler import GTReference


class OfficialReferenceBuilderTest(unittest.TestCase):
    def test_native_zero_and_other_are_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            habitat = root / "official_scene" / "habitat"
            habitat.mkdir(parents=True)
            vertex = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (2.0, 1.0, 0.0),
                    (3.0, 0.0, 0.0),
                    (4.0, 0.0, 0.0),
                    (3.0, 1.0, 0.0),
                ],
                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
            )
            face = np.empty(
                3,
                dtype=[("vertex_indices", "O"), ("object_id", "i4")],
            )
            face[0] = (np.array([0, 1, 2, 3], dtype=np.int32), 0)
            face[1] = (np.array([1, 4, 5, 2], dtype=np.int32), 7)
            face[2] = (np.array([6, 7, 8], dtype=np.int32), 9)
            PlyData(
                [
                    PlyElement.describe(vertex, "vertex"),
                    PlyElement.describe(face, "face"),
                ],
                text=True,
            ).write(str(habitat / "mesh_semantic.ply"))
            (habitat / "info_semantic.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {"id": 0, "class_name": "wall"},
                            {"id": 7, "class_name": "other"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "reference"
            build_reference(root / "official_scene", "scene", output)
            manifest = json.loads((output / "manifest.json").read_text())
            reference = GTReference.load(output / "reference.npz", manifest)
            self.assertEqual(reference.class_name_for_instance(0), "wall")
            self.assertEqual(reference.class_name_for_instance(7), "other")
            self.assertIn(0, np.unique(reference.instance))
            self.assertIn(7, np.unique(reference.instance))
            self.assertEqual(reference.instance[6:9].tolist(), [-1, -1, -1])
            self.assertEqual(manifest["unlabeled_face_object_ids"], [9])
            self.assertEqual(
                manifest["diagnostics"]["face_with_unknown_object_id_count"], 1
            )


if __name__ == "__main__":
    unittest.main()
