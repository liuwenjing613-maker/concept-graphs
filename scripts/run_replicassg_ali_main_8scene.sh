#!/usr/bin/env bash
set -u
set -o pipefail

project_root=/home/chenkejun/beauty/conceptgraphs
evaluator="$project_root/scripts/eval_replicassg_main.py"
python_bin="$project_root/envs/cg-main/bin/python"
dataset_root=/data/chenkejun/ReplicaSSG/Replica
annotations_dir=/data/chenkejun/ReplicaSSG/files
clip_cache="$project_root/models/huggingface/hub"
empty_list="$project_root/scripts/empty_replicassg_list.json"
empty_stage7="$project_root/scripts/empty_replicassg_inputs.json"
output_root="$project_root/results/ali-my/paper_main_aligned_20260821"
status_file="$output_root/replicassg_status.tsv"
gpu_index=1
object_device=${OBJECT_DEVICE:-cpu}

scenes=(room0 room1 room2 office0 office1 office2 office3 office4)

mkdir -p "$output_root"
printf 'timestamp\tmethod\tscene\tstatus\tresult\n' > "$status_file"

wait_for_gpu() {
    local method=$1
    local scene=$2
    if [[ "$object_device" == "cpu" ]]; then
        return
    fi
    while true; do
        local memory_used
        local utilization
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_index" | tr -d ' ')
        utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_index" | tr -d ' ')
        if (( memory_used <= 30000 && utilization <= 35 )); then
            return
        fi
        printf '%s\t%s\t%s\tWAIT_GPU\tmemory=%sMiB util=%s%%\n' "$(date --iso-8601=seconds)" "$method" "$scene" "$memory_used" "$utilization" >> "$status_file"
        sleep 30
    done
}

run_one() {
    local method=$1
    local scene=$2
    local ssg_scene=$3
    local map_pickle=$4
    local output_dir="$output_root/replicassg_${method}_8scene/$ssg_scene"
    local result="$output_dir/results.json"
    local visible_device=""
    if [[ "$object_device" != "cpu" ]]; then
        visible_device="$gpu_index"
    fi

    if [[ -s "$result" ]]; then
        printf '%s\t%s\t%s\tSKIP_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$method" "$scene" "$result" >> "$status_file"
        return
    fi

    wait_for_gpu "$method" "$scene"
    printf '%s\t%s\t%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$method" "$scene" "$map_pickle" >> "$status_file"

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$visible_device" nice -n 5 "$python_bin" "$evaluator" \
        --scene "$ssg_scene" \
        --dataset-root "$dataset_root" \
        --annotations-dir "$annotations_dir" \
        --map-pickle "$map_pickle" \
        --scene-graph "$empty_list" \
        --relations "$empty_list" \
        --stage7-audit "$empty_stage7" \
        --output-dir "$output_dir" \
        --clip-cache-dir "$clip_cache" \
        --device "$object_device" \
        > "$output_root/replicassg_${method}_${scene}.log" 2>&1
    local status=$?

    if (( status != 0 )); then
        printf '%s\t%s\t%s\tFAILED\texit=%s\n' "$(date --iso-8601=seconds)" "$method" "$scene" "$status" >> "$status_file"
        exit "$status"
    fi
    if [[ ! -s "$result" ]]; then
        printf '%s\t%s\t%s\tFAILED\tmissing_result\n' "$(date --iso-8601=seconds)" "$method" "$scene" >> "$status_file"
        exit 4
    fi
    printf '%s\t%s\t%s\tCOMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$method" "$scene" "$result" >> "$status_file"
}

for scene in "${scenes[@]}"; do
    ssg_scene=${scene/room/room_}
    ssg_scene=${ssg_scene/office/office_}
    suffix="ali_my_paper_main_aligned_${scene}_400f_stride5_bff233f_20260821"
    ali_map="$project_root/data/Replica/$scene/exps/$suffix/pcd_${suffix}.pkl.gz"
    run_one ali "$scene" "$ssg_scene" "$ali_map"
done

main_name=full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz
for scene in "${scenes[@]}"; do
    ssg_scene=${scene/room/room_}
    ssg_scene=${ssg_scene/office/office_}
    main_map="$project_root/results/main/replica_stride5/Replica/$scene/pcd_saves/$main_name"
    run_one main "$scene" "$ssg_scene" "$main_map"
done

printf '%s\tALL\tALL\tCOMPLETE\t16 evaluations\n' "$(date --iso-8601=seconds)" >> "$status_file"
