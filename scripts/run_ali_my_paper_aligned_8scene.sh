#!/usr/bin/env bash
set -u
set -o pipefail

project_root=/home/chenkejun/beauty/conceptgraphs
code_root="$project_root/code/official/ali-my"
python_bin="$project_root/envs/cg-ali/bin/python"
dataset_root="$project_root/data/Replica"
dataset_config="$code_root/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
log_root="$project_root/logs/ali-my/paper_main_aligned_20260821"
status_file="$log_root/status.tsv"
gpu_index=1
max_idle_memory_mib=30000
min_free_disk_gib=20
commit_tag=bff233f

scenes=(office1 office4 room1 room2 office2 office3 room0 office0)

mkdir -p "$log_root"
if [[ ! -f "$status_file" ]]; then
    printf 'timestamp\tscene\tstatus\tdetail\n' > "$status_file"
fi

cd "$code_root" || exit 2

actual_head=$(git rev-parse HEAD)
printf '%s\tMETA\tHEAD\t%s\n' "$(date --iso-8601=seconds)" "$actual_head" >> "$status_file"

for scene in "${scenes[@]}"; do
    suffix="ali_my_paper_main_aligned_${scene}_400f_stride5_${commit_tag}_20260821"
    detection_suffix="ali_my_paper_main_aligned_${scene}_detections_400f_stride5_${commit_tag}_20260821"
    output_dir="$dataset_root/$scene/exps/$suffix"
    output_pcd="$output_dir/pcd_${suffix}.pkl.gz"
    scene_log="$log_root/${scene}.log"

    if [[ -s "$output_pcd" ]]; then
        printf '%s\t%s\tSKIP_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$scene" "$output_pcd" >> "$status_file"
        continue
    fi

    while true; do
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_index" | tr -d ' ')
        utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_index" | tr -d ' ')
        free_disk_gib=$(df -BG --output=avail "$project_root" | tail -1 | tr -dc '0-9')
        if (( free_disk_gib < min_free_disk_gib )); then
            printf '%s\t%s\tSTOP_LOW_DISK\tfree=%sGiB\n' "$(date --iso-8601=seconds)" "$scene" "$free_disk_gib" >> "$status_file"
            exit 3
        fi
        if (( memory_used <= max_idle_memory_mib && utilization <= 35 )); then
            break
        fi
        printf '%s\t%s\tWAIT_GPU\tmemory=%sMiB util=%s%%\n' "$(date --iso-8601=seconds)" "$scene" "$memory_used" "$utilization" >> "$status_file"
        sleep 30
    done

    printf '%s\t%s\tSTART\tgpu=%s evidence=false stride=5\n' "$(date --iso-8601=seconds)" "$scene" "$gpu_index" >> "$status_file"

    HF_HOME="$project_root/models/huggingface" \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONHASHSEED=0 \
    PYTHONPATH=. \
    nice -n 5 "$python_bin" conceptgraph/slam/rerun_realtime_mapping.py \
        dataset_root="$dataset_root" \
        dataset_config="$dataset_config" \
        scene_id="$scene" \
        start=0 \
        end=2000 \
        stride=5 \
        make_edges=false \
        use_rerun=false \
        save_rerun=false \
        force_detection=true \
        save_detections=false \
        detections_exp_suffix="$detection_suffix" \
        exp_suffix="$suffix" \
        save_video=false \
        save_objects_all_frames=false \
        save_pcd=true \
        save_json=true \
        periodically_save_pcd=false \
        save_evidence=false \
        evidence_mode=strict \
        evidence_save_observation_pcd=false \
        save_parity_trace=true \
        seed=0 \
        device=cuda \
        hydra.verbose=false \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        > "$scene_log" 2>&1
    run_status=$?

    if (( run_status != 0 )); then
        printf '%s\t%s\tFAILED\texit=%s log=%s\n' "$(date --iso-8601=seconds)" "$scene" "$run_status" "$scene_log" >> "$status_file"
        exit "$run_status"
    fi

    if [[ ! -s "$output_pcd" ]]; then
        printf '%s\t%s\tFAILED\tmissing_output=%s\n' "$(date --iso-8601=seconds)" "$scene" "$output_pcd" >> "$status_file"
        exit 4
    fi

    output_hash=$(sha256sum "$output_pcd" | cut -d' ' -f1)
    printf '%s\t%s\tCOMPLETE\tsha256=%s path=%s\n' "$(date --iso-8601=seconds)" "$scene" "$output_hash" "$output_pcd" >> "$status_file"
done

printf '%s\tALL\tCOMPLETE\t8 scenes\n' "$(date --iso-8601=seconds)" >> "$status_file"
