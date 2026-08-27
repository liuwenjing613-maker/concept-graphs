#!/usr/bin/env python3
"""Render one ReplicaSSG trajectory into ConceptGraphs' Replica RGB-D layout.

Only RGB, metric depth, and camera poses are exported. ReplicaSSG object and
relationship annotations are deliberately not opened by this program.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import habitat_sim
import numpy as np
import quaternion as qt
from habitat_sim.utils.common import quat_from_two_vectors
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import tqdm


def _load_settings_module(replica_ssg_root: Path):
    sys.path.insert(0, str(replica_ssg_root))
    from settings import default_agent_config, default_sim_settings, make_cfg

    return default_sim_settings, make_cfg, default_agent_config


def _trajectory_samples(
    trajectory_path: Path,
    scene: str,
    fps: int,
    duration_seconds: int,
) -> tuple[np.ndarray, Rotation, np.ndarray]:
    with trajectory_path.open(encoding="utf-8") as handle:
        trajectories = json.load(handle)
    keyframes = trajectories.get(scene)
    if not isinstance(keyframes, list) or len(keyframes) < 2:
        raise ValueError(f"trajectory {trajectory_path} has no usable scene {scene}")

    translations = np.asarray(
        [frame["translation"] for frame in keyframes], dtype=np.float64
    )
    # ReplicaSSG stores quaternions as [w, x, y, z]. SciPy consumes [x, y, z, w].
    rotations_wxyz = np.asarray(
        [frame["rotation"] for frame in keyframes], dtype=np.float64
    )
    translation_amount = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    rotation_amount = np.asarray(
        [
            2.0
            * np.arccos(
                np.clip(
                    np.abs(np.dot(rotations_wxyz[index], rotations_wxyz[index + 1])),
                    -1.0,
                    1.0,
                )
            )
            for index in range(len(rotations_wxyz) - 1)
        ]
    )
    segment_lengths = translation_amount + 0.5 * rotation_amount
    total_frames = fps * duration_seconds
    frames_per_segment = np.round(
        segment_lengths / np.sum(segment_lengths) * total_frames
    ).astype(int)
    frames_per_segment[frames_per_segment == 0] = 1
    frame_times = np.concatenate([[0], np.cumsum(frames_per_segment)])

    translation_spline = CubicSpline(frame_times, translations)
    rotations_xyzw = np.concatenate(
        [rotations_wxyz[:, 1:], rotations_wxyz[:, :1]], axis=1
    )
    rotation_spline = RotationSpline(
        frame_times, Rotation.from_quat(rotations_xyzw)
    )
    sample_times = np.arange(frame_times[-1], dtype=np.int64)
    return (
        translation_spline(sample_times),
        rotation_spline(sample_times),
        frame_times,
    )


def _replica_c2w(position: np.ndarray, rotation: qt.quaternion) -> np.ndarray:
    """Match the coordinate conversion used by ReplicaSSG extract_path.py."""
    unit_y = np.eye(3, dtype=np.float32)[1]
    unit_z = np.eye(3, dtype=np.float32)[2]

    hsim_to_replica_rotation = quat_from_two_vectors(-unit_z, -unit_y)
    hsim_to_replica = np.eye(4, dtype=np.float32)
    hsim_to_replica[:3, :3] = qt.as_rotation_matrix(hsim_to_replica_rotation)

    replica_camera_to_hsim_camera_rotation = quat_from_two_vectors(-unit_z, unit_z)
    replica_camera_to_hsim_camera = np.eye(4, dtype=np.float32)
    replica_camera_to_hsim_camera[:3, :3] = qt.as_rotation_matrix(
        replica_camera_to_hsim_camera_rotation
    )

    # The upstream script names this first transform w2c, although Habitat's
    # state is camera-to-world. Preserve its two inversions exactly; the final
    # matrix is the c2w pose consumed by ConceptGraphs' ReplicaDataset.
    habitat_c2w = np.eye(4, dtype=np.float32)
    habitat_c2w[:3, :3] = qt.as_rotation_matrix(rotation)
    habitat_c2w[:3, 3] = position
    converted_inverse = (
        replica_camera_to_hsim_camera
        @ np.linalg.inv(habitat_c2w)
        @ hsim_to_replica
    )
    return np.linalg.inv(converted_inverse).astype(np.float32)


def _write_pose_file(path: Path, poses: list[np.ndarray]) -> None:
    temporary = path.with_suffix(".txt.incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for pose in poses:
            handle.write(" ".join(f"{float(value):.9g}" for value in pose.reshape(-1)))
            handle.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replica-ssg-root", type=Path, default=Path("/data/chenkejun/ReplicaSSG")
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/chenkejun/ReplicaSSG/ConceptGraphs"),
    )
    parser.add_argument("--scene", default="room_0")
    parser.add_argument("--sequence", default="room0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Render only the first N interpolated frames (for smoke tests)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    replica_ssg_root = args.replica_ssg_root.resolve()
    replica_data_root = replica_ssg_root / "Replica" / "data"
    trajectory_path = replica_ssg_root / "files" / "trajectories.json"
    scene_mesh = replica_data_root / args.scene / "habitat" / "mesh_semantic.ply"
    scene_dataset_config = replica_data_root / "replica.scene_dataset_config.json"
    for required in (trajectory_path, scene_mesh, scene_dataset_config):
        if not required.is_file():
            raise FileNotFoundError(required)

    positions, rotations, frame_times = _trajectory_samples(
        trajectory_path, args.scene, args.fps, args.duration_seconds
    )
    available_frames = len(positions)
    render_count = (
        min(args.max_frames, available_frames)
        if args.max_frames is not None
        else available_frames
    )
    if render_count < 1:
        raise ValueError("max-frames must be positive")

    output_scene = args.dataset_root.resolve() / args.sequence
    results_dir = output_scene / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    incomplete_marker = output_scene / "RENDER_INCOMPLETE"
    incomplete_marker.write_text(f"started_at_unix={time.time()}\n", encoding="utf-8")
    ready_marker = output_scene / "READY"
    if ready_marker.exists():
        ready_marker.unlink()

    default_sim_settings, make_cfg, default_agent_config = _load_settings_module(
        replica_ssg_root
    )
    sim_settings = dict(default_sim_settings)
    sim_settings.update(
        {
            "scene": args.scene,
            "scene_dataset_config_file": str(scene_dataset_config),
            "width": args.width,
            "height": args.height,
            "hfov": args.hfov,
            "sensor_height": 0,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "silent": True,
        }
    )
    config = make_cfg(sim_settings)
    config.sim_cfg.gpu_device_id = args.gpu
    agent_id = sim_settings["default_agent"]
    config.agents[agent_id] = default_agent_config(config, agent_id)
    simulator = habitat_sim.Simulator(config)
    agent = simulator.initialize_agent(agent_id)

    poses: list[np.ndarray] = []
    try:
        for frame_index in tqdm(range(render_count), desc=f"render {args.sequence}"):
            xyzw = rotations[frame_index].as_quat()
            rotation = qt.quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2])
            position = positions[frame_index]
            state = agent.get_state()
            state.position = position
            state.rotation = rotation
            for sensor_state in state.sensor_states.values():
                sensor_state.position = position
                sensor_state.rotation = rotation
            agent.set_state(state)

            observations = simulator.get_sensor_observations()
            color = np.asarray(observations["color_sensor"])[..., :3].astype(np.uint8)
            depth_m = np.asarray(observations["depth_sensor"], dtype=np.float32)
            depth_mm = np.rint(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
            Image.fromarray(color, mode="RGB").save(
                results_dir / f"frame{frame_index:06d}.jpg", quality=95
            )
            Image.fromarray(depth_mm, mode="I;16").save(
                results_dir / f"depth{frame_index:06d}.png"
            )
            poses.append(_replica_c2w(position, rotation))
    finally:
        simulator.close()

    _write_pose_file(output_scene / "traj.txt", poses)
    focal_length = (args.width / 2.0) / np.tan(np.deg2rad(args.hfov / 2.0))
    metadata = {
        "schema_version": "0.1.0",
        "source": "ReplicaSSG official room trajectory rendered with Habitat-Sim",
        "source_scene": args.scene,
        "sequence": args.sequence,
        "frame_count": render_count,
        "official_interpolated_frame_count": available_frames,
        "official_frame_times_end": int(frame_times[-1]),
        "width": args.width,
        "height": args.height,
        "hfov_degrees": args.hfov,
        "fx": focal_length,
        "fy": focal_length,
        "cx": args.width / 2.0,
        "cy": args.height / 2.0,
        "png_depth_scale": 1000.0,
        "object_annotations_loaded": False,
        "relationship_annotations_loaded": False,
        "semantic_sensor_enabled": False,
        "trajectory_noise_enabled": False,
        "finished_at_unix": time.time(),
    }
    with (output_scene / "render_metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    incomplete_marker.unlink(missing_ok=True)
    ready_marker.write_text("ready\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
