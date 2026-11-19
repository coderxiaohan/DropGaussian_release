#!/usr/bin/env bash
set -euo pipefail

# Common original DropGaussian / direct MVS + VCD/VCP runner for one setting.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
python_bin=${PYTHON_BIN:-python}
benchmark_mode=${MODE:-mvs}
dataset=${DATASET:-llff}
gpu_id=${GPU_ID:-0}
seed=${SEED:-0}
port=${PORT:-0}

for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        printf '%s\n' \
            'Usage: MODE=mvs DATASET=llff bash scripts/all/train_mvsdrop_benchmark.sh [train.py arguments...]' \
            'MODE: mvs, dropgaussian, or both; configure DATASET, N_VIEWS, SCENES, GPU_ID and OUTPUT_ROOT.' \
            'Every run requires a fresh or empty output setting directory, including baseline runs.' \
            'Existing nonempty directories, files and symlinks are rejected before training, rendering or summary writes.' \
            'MODE=both checks both setting directories before either method starts.' \
            'Use a new OUTPUT_ROOT for repeats; historical results are never reused under a new pruning policy.'
        exit 0
    fi
done

# All policies use this check, including CLI density/strategy overrides. This
# prevents legacy result folders from being reused under the current algorithm.
require_fresh_output() (
    setting_output=$1
    if [[ -e "$setting_output" || -L "$setting_output" ]]; then
        if [[ -L "$setting_output" || ! -d "$setting_output" ]]; then
            echo "Output setting must be a real directory: $setting_output" >&2
            exit 2
        fi
        shopt -s nullglob dotglob
        output_entries=("$setting_output"/*)
        if (( ${#output_entries[@]} )); then
            echo "Output setting directory is nonempty: $setting_output" >&2
            echo "Use a fresh OUTPUT_ROOT; existing results cannot be reused under a different training policy." >&2
            exit 2
        fi
    fi
)

if [[ "$benchmark_mode" != "dropgaussian" && "$benchmark_mode" != "mvs" && "$benchmark_mode" != "both" ]]; then
    echo "MODE must be dropgaussian, mvs, or both; adaptive mvsdrop is not available in this branch (got: $benchmark_mode)." >&2
    exit 2
fi

# This runner owns the initializer. Remove conflicting values before argparse:
# a forbidden earlier value would fail validation even when a later one wins.
train_args=()
while (( $# )); do
    case "$1" in
        --init_mode|--mvs_init_strategy)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            shift 2
            ;;
        --init_mode=*|--mvs_init_strategy=*) shift ;;
        *) train_args+=("$1"); shift ;;
    esac
done

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

if [[ "$benchmark_mode" == "both" ]]; then
    paired_root=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/paired}
    for paired_mode in dropgaussian mvs; do
        require_fresh_output "$paired_root/$paired_mode/${dataset}${n_views}"
    done
    for paired_mode in dropgaussian mvs; do
        MODE="$paired_mode" OUTPUT_ROOT="$paired_root/$paired_mode/${dataset}${n_views}" \
            bash "$script_dir/train_mvsdrop_benchmark.sh" "${train_args[@]}"
    done
    exit 0
fi

extra_args=(--density_control "${DENSITY_CONTROL:-none}")
forced_init_args=(--init_mode colmap_dense)
if [[ "$benchmark_mode" == "mvs" ]]; then
    # Direct replacement only. Point-count capping is a resource budget, not a
    # reliability gate, residual fusion, or confidence-conditioned attribute.
    extra_args=(--density_control "${DENSITY_CONTROL:-vcd_vcp}"
        --mvs_init_max_points "${MVS_INIT_MAX_POINTS:-100000}")
    forced_init_args=(--init_mode mvs_anywhere --mvs_init_strategy replace)
fi
if [[ "$benchmark_mode" != "dropgaussian" && -n "${MVS_INIT_PATH:-}" ]]; then
    extra_args+=(--mvs_init_path "$MVS_INIT_PATH")
fi

read -r -a scenes <<< "${SCENES:-$default_scenes}"
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/final/$benchmark_mode/${dataset}${n_views}}
require_fresh_output "$output_root"
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
            "${extra_args[@]}" "${train_args[@]}" "${forced_init_args[@]}" \
            2>&1 | tee "$run_dir/train.log"
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
            "$script_dir/evaluate_dtu.py" --model-path "$run_dir" \
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

"$python_bin" "$script_dir/summarize_mvsdrop_setting.py" \
    --experiment-root "$output_root" --iteration "$iterations"
echo "$benchmark_mode summary: $output_root/metrics_mean.txt"
