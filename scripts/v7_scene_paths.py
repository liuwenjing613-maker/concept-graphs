"""Keep raw Replica inputs separate from fresh experiment outputs."""
import re
from pathlib import Path


def scene_view(source_root, output_root, scene, *, create=False):
    if not re.fullmatch(r"(?:room|office)[0-9]+", scene):
        raise ValueError("scene must be a Replica scene name, e.g. room0 or office2")
    source = Path(source_root).resolve() / scene
    output_root = Path(output_root).resolve()
    target = output_root / scene
    if target.is_symlink():
        raise ValueError(f"output scene must not redirect outside the output root: {target}")
    if (target / "exps").is_symlink():
        raise ValueError(f"output exps must not be a symlink: {target / 'exps'}")
    for name in ("results", "traj.txt"):
        src, dst = source / name, target / name
        valid = src.is_dir() if name == "results" else src.is_file()
        if not valid:
            raise FileNotFoundError(f"Replica input missing: {src}")
        if dst.exists() or dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"existing input differs; refusing to replace: {dst}")
    if not any((source / "results").glob("frame*.jpg")) or not any((source / "results").glob("depth*.png")):
        raise FileNotFoundError(f"Replica RGB/depth frames missing: {source / 'results'}")
    if create:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("results", "traj.txt"):
            dst = target / name
            if not dst.exists():
                try:
                    dst.symlink_to((source / name).resolve(), target_is_directory=name == "results")
                except FileExistsError:
                    if dst.resolve() != (source / name).resolve():
                        raise
    return output_root
