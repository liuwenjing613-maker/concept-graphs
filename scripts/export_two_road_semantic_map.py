#!/usr/bin/env python3
"""Export a bbox semantic evaluation copy without modifying the online map."""
import argparse
import gzip
import pickle
from pathlib import Path
import numpy as np


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError('evaluation export must use a new destination')
    with gzip.open(source, 'rb') as stream:
        data = pickle.load(stream)
    for obj in data['objects']:
        semantic = obj['clip_semantic_ft']
        if hasattr(semantic, 'detach'): semantic = semantic.detach().cpu().numpy()
        semantic = np.asarray(semantic)
        if semantic.shape != (1024,) or not np.isfinite(semantic).all():
            raise ValueError('invalid bbox semantic feature')
        obj['clip_ft'] = semantic.copy()
    data['two_road_readout'] = {'source_map': str(source.resolve()),
        'feature': 'clip_semantic_ft', 'evaluation_only': True,
        'geometry_and_membership_unchanged': True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, 'wb') as stream:
        pickle.dump(data, stream)
    return len(data['objects'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('destination')
    args = parser.parse_args()
    print('exported objects:', export(args.source, args.destination))
