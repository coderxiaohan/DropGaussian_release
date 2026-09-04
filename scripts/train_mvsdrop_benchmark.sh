#!/usr/bin/env bash
set -euo pipefail

# Matched DropGaussian / MVSInit-DG runner for one dataset-view setting.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
benchmark_mode=${MODE:-mvsdrop}
dataset=${DATASET:-llff}
gpu_id=${GPU_ID:-0}
seed=${SEED:-0}
port=${PORT:-0}

if [[ "$benchmark_mode" != "dropgaussian" && "$benchmark_mode" != "mvsdrop" ]]; then
    echo "MODE must be dropgaussian or mvsdrop, got: $benchmark_mode" >&2
    exit 2
fi

dataset_args=()
case "$dataset" in
    llff)
        data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/nerf_llff_data}
        n_views=${N_VIEWS:-3}
        resolution=${RESOLUTION:-8}
        iterations=${ITERATIONS:-10000}
        default_scenes="fern flower fortress horns leaves orchids room trex"
        dataset_args+=(--dataset_name llff)
        ;;
    dtu)
        data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/DTU_colmap}
        n_views=${N_VIEWS:-3}
        resolution=${RESOLUTION:-4}
        iterations=${ITERATIONS:-10000}
        default_scenes="scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41 scan45 scan55 scan63 scan82 scan103 scan110 scan114"
        mask_root=${DTU_MASK_ROOT:-$data_root/idrmasks}
        dataset_args+=(--dataset_name dtu)
        ;;
    mipnerf360)
        data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/mipnerf360}
        n_views=${N_VIEWS:-12}
        resolution=${RESOLUTION:-8}
        iterations=${ITERATIONS:-10000}
        default_scenes="bicycle bonsai counter garden kitchen room stump"
        dataset_args+=(--dataset_name mipnerf360)
        ;;
    blender)
        data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/nerf_synthetic}
        n_views=${N_VIEWS:-8}
        resolution=${RESOLUTION:-1}
        iterations=${ITERATIONS:-10000}
        default_scenes="chair drums ficus hotdog lego materials mic ship"
        dataset_args+=(--dataset_name blender --white_background --rand_pcd)
        ;;
    *)
        echo "DATASET must be llff, dtu, mipnerf360, or blender, got: $dataset" >&2
        exit 2
        ;;
esac

extra_args=(--init_mode colmap_dense)
if [[ "$benchmark_mode" == "mvsdrop" ]]; then
    extra_args=(
        --init_mode mvs_anywhere
        --mvs_init_max_points "${MVS_INIT_MAX_POINTS:-250000}"
        --mvs_init_strategy "${MVS_INIT_STRATEGY:-auto}"
        --mvs_init_total_max_points "${MVS_INIT_TOTAL_MAX_POINTS:-350000}"
        --mvs_init_dense_base_threshold "${MVS_INIT_DENSE_BASE_THRESHOLD:-200000}"
        --mvs_init_dense_residual_points "${MVS_INIT_DENSE_RESIDUAL_POINTS:-100000}"
    )
    if [[ -n "${MVS_INIT_PATH:-}" ]]; then
        extra_args+=(--mvs_init_path "$MVS_INIT_PATH")
    fi
fi

read -r -a scenes <<< "${SCENES:-$default_scenes}"
output_root=${OUTPUT_ROOT:-$repo_root/output/mvsdrop/final/$benchmark_mode/${dataset}${n_views}}
mkdir -p "$output_root"

test_iterations=()
for percent in 5 10 20 40 70 100; do
    checkpoint=$(( (iterations * percent + 99) / 100 ))
    if [[ ${#test_iterations[@]} -eq 0 || ${test_iterations[-1]} -ne $checkpoint ]]; then
        test_iterations+=("$checkpoint")
    fi
done

for scene in "${scenes[@]}"; do
    if [[ "$dataset" == "dtu" ]]; then
        scene=${scene%_colmap}
        scene_root="$data_root/${scene}_colmap"
    else
        scene_root="$data_root/$scene"
    fi
    run_dir="$output_root/$scene"
    final_ply="$run_dir/point_cloud/iteration_${iterations}/point_cloud.ply"
    metrics_file="$run_dir/metrics_${iterations}.txt"
    [[ -d "$scene_root" ]] || { echo "Missing scene: $scene_root" >&2; exit 2; }
    mkdir -p "$run_dir"

    if [[ ! -f "$final_ply" || ! -f "$run_dir/manifest.json" ]]; then
        echo "Training $benchmark_mode on $dataset/$scene ($n_views views)..."
        CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/train.py" \
            -s "$scene_root" -m "$run_dir" "${dataset_args[@]}" \
            --eval -r "$resolution" --n_views "$n_views" \
            --iterations "$iterations" --test_iterations "${test_iterations[@]}" \
            --save_iterations "$iterations" --seed "$seed" --port "$port" \
            "${extra_args[@]}" "$@" 2>&1 | tee "$run_dir/train.log"
    else
        echo "Skipping completed training for $dataset/$scene."
    fi

    if [[ "$dataset" == "dtu" && ! -d "$run_dir/test/ours_${iterations}/renders" ]] \
        || [[ "$dataset" != "dtu" && ! -f "$metrics_file" ]]; then
        echo "Rendering $dataset/$scene..."
        CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/render.py" \
            -m "$run_dir" -r "$resolution" --iteration "$iterations" \
            --skip_train 2>&1 | tee "$run_dir/render_${iterations}.log"
    fi

    if [[ "$dataset" == "dtu" ]]; then
        echo "Evaluating DTU/$scene with IDR masks..."
        CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" \
            "$repo_root/scripts/evaluate_dtu.py" --model-path "$run_dir" \
            --mask-root "$mask_root" --scene "$scene" --iteration "$iterations" \
            2>&1 | tee "$run_dir/evaluate_${iterations}.log"
    else
        for _ in {1..10}; do
            [[ -f "$metrics_file" ]] && break
            sleep 1
        done
        if [[ ! -f "$metrics_file" ]]; then
            echo "Expected render.py to create $metrics_file" >&2
            exit 2
        fi
    fi
done

"$python_bin" "$repo_root/scripts/summarize_mvsdrop_setting.py" \
    --experiment-root "$output_root" --iteration "$iterations"
echo "$benchmark_mode summary: $output_root/metrics_mean.txt"
