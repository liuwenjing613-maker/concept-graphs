"""Explicit readout exports; association always keeps clip_ft unchanged."""
from pathlib import Path
import json
import numpy as np
from conceptgraph.slam.slam_classes import to_numpy


def export_readouts(objects, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def stack(key):
        rows = [np.asarray(to_numpy(obj[key]), dtype=np.float32) for obj in objects]
        values = np.stack(rows) if rows else np.empty((0, 1024), dtype=np.float32)
        if values.shape != (len(objects), 1024) or not np.isfinite(values).all():
            raise ValueError('invalid readout features: '+key)
        return values
    association = stack('clip_ft')
    semantic = stack('clip_semantic_ft')
    np.savez_compressed(output_dir/'clip_readouts.npz', association_fused=association, semantic_bbox=semantic)
    (output_dir/'clip_readouts.json').write_text(json.dumps({
        'schema': 'v7-clip-two-road-v1',
        'object_ids': [str(obj['id']) for obj in objects],
        'observation_uids': [obj.get('obs_uids', []) for obj in objects],
        'num_detections': [int(obj['num_detections']) for obj in objects],
        'association_feature': 'clip_ft', 'semantic_feature': 'clip_semantic_ft',
        'aggregation': 'normalized recursive merge weighted by accepted detection count',
        'same_map_same_membership': True,
    }, indent=2)+'\n')
