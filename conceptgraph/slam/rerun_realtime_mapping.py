'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import json
import random
import uuid
from pathlib import Path
import pickle
import gzip

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client, gpt_model
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
from conceptgraph.utils.evidence import EvidenceRecorder
from conceptgraph.slam.association_gate import BlockingAssociationGate, DISCARD_MATCH_INDEX

from conceptgraph.revision.corruption import (
    ControlledCorruptionController,
    load_corruption_plan,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    rerun_connect_addr = cfg.get("rerun_connect_addr")
    if cfg.use_rerun and rerun_connect_addr:
        orr.connect_grpc(str(rerun_connect_addr))
    else:
        orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM('sam_l.pt') # SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    # The no-edge Smoke Test must not require credentials or make VLM calls.
    # The historical code initialized the client unconditionally and later
    # consolidated captions even when make_edges was false.
    openai_client = get_openai_client() if cfg.make_edges else None

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    evidence = EvidenceRecorder(
        exp_out_path=exp_out_path,
        cfg=cfg,
        detection_cfg=detections_exp_cfg,
        enabled=bool(getattr(cfg, "save_evidence", True)),
        model_versions={
            "detector": "yolov8l-world.pt",
            "segmenter": "sam_l.pt",
            "clip": "ViT-H-14/laion2b_s32b_b79k",
            "vlm": gpt_model,
        },
        prompt_versions={
            "FRAME_EDGE": "ali-dev-system_prompt_only_top-v1",
            "FRAME_CAPTION": "ali-dev-system_prompt_captions-v1",
            "OBJECT_CAPTION_CONSOLIDATION": "ali-dev-system_prompt_consolidate_captions-v1",
        },
    )
    openai_client = evidence.wrap_openai_client(openai_client)
    association_gate = BlockingAssociationGate(
        cfg=cfg,
        output_dir=exp_out_path / "blocking_association_gate",
        rerun=orr,
    )
    parity_trace = []
    revision_cfg = cfg.get("revision") or {}
    corruption_controller = None
    if bool(revision_cfg.get("enabled", False)):
        if str(revision_cfg.get("mode")) != "controlled_validation":
            raise ValueError("v0 live revision only supports controlled_validation mode")
        corruption_plan_path = revision_cfg.get("corruption_plan")
        if not corruption_plan_path:
            raise ValueError("revision.enabled requires revision.corruption_plan")
        corruption_plan = load_corruption_plan(corruption_plan_path)
        corruption_controller = ControlledCorruptionController(
            corruption_plan,
            output_dir=(
                exp_out_path
                / str(revision_cfg.get("log_dir_name", "revision"))
                / corruption_plan.case_uid
            ),
            require_exactly_once=True,
        )

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        frame_uid = evidence.frame_uid(frame_idx)
        color_path = Path(dataset.color_paths[frame_idx])
        depth_paths = getattr(dataset, "depth_paths", None)
        depth_path = Path(depth_paths[frame_idx]) if depth_paths is not None else None

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            evidence.record_frame(
                frame_idx=frame_idx,
                source_frame_id=color_path.stem,
                rgb_path=color_path,
                depth_path=depth_path,
                pose=None,
                intrinsics=None,
                processed=False,
                skip_reason="early_exit",
                num_raw_detections=0,
                num_kept_observations=0,
            )
            continue

        # Read info about current frame from dataset
        # color image
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / color_path.stem
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data

                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # Make the edges
            evidence.begin_vlm_context(
                frame_uid=frame_uid,
                input_image_ref=str(vis_save_path_for_vlm),
                input_labels=detection_class_labels,
                input_observation_uids=[
                    evidence.observation_uid(frame_idx, index)
                    for index in range(len(curr_det.xyxy))
                ],
            )
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, openai_client)
            evidence.finish_vlm_context(
                {"FRAME_EDGE": edges, "FRAME_CAPTION": captions}
            )

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                # if no detections, throw an error
                raise FileNotFoundError(f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / color_path.stem} or \n{det_exp_pkl_path / f'{int(color_path.stem):06}'}.")

        # Stable raw detection identities are attached before resize/filter.
        raw_observation_snapshots = evidence.prepare_observations(
            raw_gobs=raw_gobs,
            frame_idx=frame_idx,
            detection_path=detections_path,
        )

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs, filter_trace = filter_gobs(resized_gobs, image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
            return_trace=True,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            evidence.record_observations(
                frame_idx=frame_idx,
                snapshots=raw_observation_snapshots,
                filtered_gobs=gobs,
                obj_pcds_and_bboxes=[],
                image_shape=image_rgb.shape,
                bg_classes=obj_classes.get_bg_classes_arr(),
                filter_trace=filter_trace,
                depth_array=depth_array,
            )
            evidence.record_filter_trace(frame_idx, filter_trace)
            evidence.record_frame(
                frame_idx=frame_idx,
                source_frame_id=color_path.stem,
                rgb_path=color_path,
                depth_path=depth_path,
                pose=adjusted_pose,
                intrinsics=intrinsics,
                processed=True,
                skip_reason="no_kept_2d_observations",
                num_raw_detections=len(raw_observation_snapshots),
                num_kept_observations=0,
            )
            continue

        # this helps make sure things like pillows on couches are separate objects
        pre_subtract_masks = np.asarray(gobs['mask'], dtype=bool).copy()
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"], obj["evidence_pcd_stats"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    return_stats=True,
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_obs_uids = evidence.record_observations(
            frame_idx=frame_idx,
            snapshots=raw_observation_snapshots,
            filtered_gobs=gobs,
            obj_pcds_and_bboxes=obj_pcds_and_bboxes,
            image_shape=image_rgb.shape,
            bg_classes=obj_classes.get_bg_classes_arr(),
            filter_trace=filter_trace,
            pre_subtract_masks=pre_subtract_masks,
            depth_array=depth_array,
        )
        evidence.record_filter_trace(frame_idx, filter_trace)

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )
        evidence.attach_observation_membership(detection_list, detection_obs_uids)
        evidence.record_frame(
            frame_idx=frame_idx,
            source_frame_id=color_path.stem,
            rgb_path=color_path,
            depth_path=depth_path,
            pose=adjusted_pose,
            intrinsics=intrinsics,
            processed=True,
            skip_reason=None if len(detection_list) else "no_kept_3d_observations",
            num_raw_detections=len(raw_observation_snapshots),
            num_kept_observations=len(detection_obs_uids),
        )

        if len(detection_list) == 0: # no detections, skip
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            initial_matches = [None] * len(detection_list)
            empty_similarity = np.empty((len(detection_list), 0), dtype=np.float32)
            evidence.record_associations(
                frame_idx, detection_list, objects,
                empty_similarity, empty_similarity, empty_similarity,
                initial_matches,
            )
            objects.extend(detection_list)
            for detected_obj_idx, created_object in enumerate(detection_list):
                evidence.record_association_object_version(
                    frame_idx=frame_idx,
                    detected_obj_idx=detected_obj_idx,
                    existing_obj_match_idx=None,
                    before_object=None,
                    after_object=created_object,
                )
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )
        baseline_match_indices = list(match_indices)
        match_indices = association_gate.route_frame(
            frame_idx=frame_idx,
            source_frame_id=color_path.stem,
            image_rgb=image_rgb,
            detection_list=detection_list,
            objects=objects,
            aggregate_sim=agg_sim,
            baseline_match_indices=baseline_match_indices,
        )
        gate_discarded_indices = {
            index for index, match_index in enumerate(match_indices)
            if match_index == DISCARD_MATCH_INDEX
        }
        if corruption_controller is not None:
            match_indices = corruption_controller.apply(
                frame_idx=frame_idx,
                detection_list=detection_list,
                objects=objects,
                original_match_indices=match_indices,
            )
            # A quality rejection is terminal for this observation.  Synthetic
            # corruption experiments must not accidentally turn it into a
            # create or merge action.
            for index in gate_discarded_indices:
                match_indices[index] = DISCARD_MATCH_INDEX
        evidence.record_associations(
            frame_idx, detection_list, objects,
            spatial_sim, visual_sim, agg_sim, match_indices,
        )

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device'],
            object_update_callback=lambda detected_obj_idx, existing_obj_match_idx, before_object, after_object: evidence.record_association_object_version(
                frame_idx=frame_idx,
                detected_obj_idx=detected_obj_idx,
                existing_obj_match_idx=existing_obj_match_idx,
                before_object=before_object,
                after_object=after_object,
            ),
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        edges_before_online_update = evidence.snapshot_edges(map_edges, objects)
        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        evidence.record_edge_diff(
            frame_idx=frame_idx,
            before=edges_before_online_update,
            map_edges=map_edges,
            objects=objects,
            reason="online_relation_update",
            source_observation_uids=detection_obs_uids,
        )
        frame_parity = {
            "frame_idx": int(frame_idx),
            "source_frame_id": color_path.stem,
            "objects_after_association": len(objects),
            "denoise": {"executed": False},
            "filter": {"executed": False},
            "merge": {"executed": False},
        }
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_before_cleanup = evidence.snapshot_edges(map_edges, objects)
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name'] 
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])
        evidence.record_edge_diff(
            frame_idx=frame_idx,
            before=edges_before_cleanup,
            map_edges=map_edges,
            objects=objects,
            reason="low_support_cleanup",
        )
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            denoise_before_count = len(objects)
            objects_before_denoise = evidence.snapshot_objects(objects)
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )
            evidence.record_denoise(frame_idx, objects_before_denoise, objects)
            frame_parity["denoise"] = {
                "executed": True,
                "objects_before": denoise_before_count,
                "objects_after": len(objects),
            }

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            filter_before_count = len(objects)
            objects_before_filter = evidence.snapshot_objects(objects)
            edges_before_filter = evidence.snapshot_edges(map_edges, objects)
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )
            evidence.record_filter(frame_idx, objects_before_filter, objects)
            evidence.record_edge_diff(
                frame_idx=frame_idx,
                before=edges_before_filter,
                map_edges=map_edges,
                objects=objects,
                reason="object_filter",
            )
            frame_parity["filter"] = {
                "executed": True,
                "objects_before": filter_before_count,
                "objects_after": len(objects),
                "removed_count": filter_before_count - len(objects),
            }

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            merge_before_count = len(objects)
            edges_before_merge = evidence.snapshot_edges(map_edges, objects)
            merge_result = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges,
                merge_event_callback=lambda source_object, target_object, overlap_ratio, visual_similarity, text_similarity: evidence.record_object_merge(
                    frame_idx=frame_idx,
                    source_object=source_object,
                    target_object=target_object,
                    overlap_ratio=overlap_ratio,
                    visual_similarity=visual_similarity,
                    text_similarity=text_similarity,
                ),
                merge_decision_callback=lambda source_object, target_object, overlap_ratio, visual_similarity, text_similarity, decision, reject_reason, source_active_before, target_active_before, candidate_rank: evidence.record_merge_candidate(
                    frame_idx=frame_idx,
                    source_object=source_object,
                    target_object=target_object,
                    overlap_ratio=overlap_ratio,
                    visual_similarity=visual_similarity,
                    text_similarity=text_similarity,
                    decision=decision,
                    reject_reason=reject_reason,
                    source_active_before=source_active_before,
                    target_active_before=target_active_before,
                ),
            )
            # ali-dev's merge_objects returns only objects when edge merging
            # is disabled, and (objects, map_edges) when it is enabled.
            if cfg["make_edges"]:
                objects, map_edges = merge_result
            else:
                objects = merge_result
            evidence.record_edge_diff(
                frame_idx=frame_idx,
                before=edges_before_merge,
                map_edges=map_edges,
                objects=objects,
                reason="object_merge",
            )
            frame_parity["merge"] = {
                "executed": True,
                "objects_before": merge_before_count,
                "objects_after": len(objects),
                "merged_count": merge_before_count - len(objects),
            }
        frame_parity["objects_final"] = len(objects)
        frame_parity["edges_final"] = len(map_edges.edges_by_index)
        parity_trace.append(frame_parity)
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.vis_render:
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True
            )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------

    if cfg.get("save_parity_trace", False):
        (exp_out_path / "parity_trace.json").write_text(
            json.dumps(parity_trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    
    # Consolidate captions only for the VLM edge/caption mode.  In the normal
    # no-edge mapping path, keep an explicit empty field without any API call.
    if cfg.make_edges:
        for object in objects:
            obj_captions = object['captions'][:20]
            evidence.begin_vlm_context(
                object_uid=str(object['id']),
                input_captions=obj_captions,
                input_observation_uids=object.get('obs_uids', []),
            )
            consolidated_caption = consolidate_captions(openai_client, obj_captions)
            evidence.finish_vlm_context(
                {"OBJECT_CAPTION_CONSOLIDATION": consolidated_caption}
            )
            object['consolidated_caption'] = consolidated_caption
    else:
        for object in objects:
            object['consolidated_caption'] = ""

    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # Save the pointcloud
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges
        )

    if cfg.save_json:
        save_obj_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        save_edge_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects,
            edges=map_edges
        )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    if corruption_controller is not None:
        corruption_controller.finalize()
    association_gate.close(
        status="early_exit" if exit_early_flag else "completed",
    )
    evidence.close(
        status="early_exit" if exit_early_flag else "completed",
        objects=objects,
        map_edges=map_edges,
    )
    owandb.finish()

if __name__ == "__main__":
    main()
