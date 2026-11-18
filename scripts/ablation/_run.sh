#!/usr/bin/env bash
set -euo pipefail

# Fixed module ablations. Reuse the existing train/render/evaluate pipeline;
# do not change the initializer, density algorithms, or complete-method launcher.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
variant=${1:?An ablation variant is required}
dataset=${2:?A dataset is required}
shift 2

complement=0
case "$variant" in
    baseline) density=none ;;
    complement_only) complement=1; density=none ;;
    egd_only) density=vcd ;;
    tbp_only) density=vcp ;;
    complement_egd) complement=1; density=vcd ;;
    complement_tbp) complement=1; density=vcp ;;
    egd_tbp) density=vcd_vcp ;;
    *) echo "Unknown ablation variant: $variant" >&2; exit 2 ;;
esac
case "$dataset" in
    llff|dtu) default_views=3 ;;
    mipnerf360) default_views=12 ;;
    blender) default_views=8 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

dry_run=${DRY_RUN:-0}
train_args=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift; continue ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/ablation/${variant}/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
                '' \
                "Fixed variant: ${variant}; geometry complement=${complement}; density_control=${density}." \
                'EGD = vcd, TBP = vcp; none retains the original DropGaussian density control.' \
                'Only the selected modules change; DropGaussian regularization remains enabled.' \
                'Complement means retaining reconstructed base geometry and adding coverage-selected MVS points,' \
                'not direct replacement by MVS. Missing reconstructed geometry follows the full-method fallback.' \
                'Blender always uses original random initialization (10000 points), without MVS caches.' \
                'On Blender: complement_only=baseline, complement_egd=egd_only, complement_tbp=tbp_only;' \
                'egd_tbp is equivalent to the current complete-method Blender policy.' \
                "Defaults: all configured ${dataset} scenes, ${default_views} training views, 10000 iterations, GPU 0." \
                'All scenes does not mean all images are used for training; N_VIEWS selects sparse views.' \
                'Overrides: N_VIEWS, GPU_ID, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION,' \
                'PYTHON_BIN, OUTPUT_ROOT, DTU_MASK_ROOT, SEED, PORT.' \
                'Real-dataset complementary variants also accept MVS_INIT_PATH and numeric MVS_COMPLEMENT_* settings:' \
                'MAX_POINTS=100000, TOTAL_POINTS=0, CANDIDATE_POINTS=500000, COVERAGE_PIXELS=2.0, DEPTH_SCALE=0.02.' \
                'TOTAL_POINTS=0 keeps all base points; a positive cap may subsample base geometry.' \
                'MVS caches must match training views; base geometry training-view-only provenance is not independently verified.' \
                'MODE, DENSITY_CONTROL and MVS_COMPLEMENT_MODE are fixed by the variant; conflicting values are rejected.' \
                'Leave those method environment variables unset. Use environment settings, not method/protocol CLI overrides.' \
                'OUTPUT_ROOT is always a common parent: <root>/<variant>/<dataset><views>/<scene>.' \
                'Default parent: <repo>/output/ablation. Existing nonempty settings are rejected, never reused.' \
                'Use a fresh OUTPUT_ROOT when changing parameters. --dry-run or DRY_RUN=1 writes nothing.' \
                'Each run performs train -> render -> evaluate -> summarize; no training is started by --help.'
            exit 0
            ;;
        --) echo "The -- option separator is not supported." >&2; exit 2 ;;
    esac

    argument_name=${1%%=*}
    case "$argument_name" in
        -m*) echo "Use OUTPUT_ROOT, not $argument_name; this launcher owns model paths." >&2; exit 2 ;;
        -s*) echo "Use DATA_ROOT and SCENES, not $argument_name; this launcher owns source paths." >&2; exit 2 ;;
        -r*) echo "Use RESOLUTION, not $argument_name; training and evaluation must agree." >&2; exit 2 ;;
        -i*|-w*) echo "Image-loading and background protocol options are fixed." >&2; exit 2 ;;
        --mvs*|--density*|--vcd*|--vcp*)
            echo "Method CLI overrides are disabled; select the appropriate ablation directory." >&2
            exit 2
            ;;
        --*)
            # argparse accepts long-option abbreviations, so reject prefixes as
            # well as full names and --option=value forms for owned settings.
            for reserved in --model_path --source_path --iterations --n_views \
                --dataset_name --resolution --mvs_complement_mode --density_control \
                --init_mode --mvs_init_strategy --rand_pcd --white_background \
                --images --eval --llffhold --seed --test_iterations --save_iterations \
                --start_checkpoint --sh_degree --position_lr_init --position_lr_final \
                --position_lr_delay_mult --position_lr_max_steps --feature_lr \
                --opacity_lr --scaling_lr --rotation_lr --percent_dense --lambda_dssim \
                --densification_interval --opacity_reset_interval --densify_from_iter \
                --densify_until_iter --densify_grad_threshold --random_background \
                --vcd_error_quantile --vcp_contribution_quantile; do
                if [[ "$reserved" == "$argument_name"* ]]; then
                    echo "This launcher owns $reserved; use documented environment settings, not $argument_name." >&2
                    exit 2
                fi
            done
            ;;
    esac
    train_args+=("$1")
    shift
done

logical_mode=dropgaussian
complement_mode=none
if [[ "$complement" == 1 ]]; then
    logical_mode=mvs
    complement_mode=coverage
fi
if [[ ${MODE+x} && "$MODE" != "$logical_mode" ]]; then
    echo "${variant} fixes MODE=${logical_mode}; unset the conflicting MODE=${MODE}." >&2
    exit 2
fi
if [[ ${DENSITY_CONTROL+x} && "$DENSITY_CONTROL" != "$density" ]]; then
    echo "${variant} fixes DENSITY_CONTROL=${density}; unset the conflicting DENSITY_CONTROL=${DENSITY_CONTROL}." >&2
    exit 2
fi
if [[ ${MVS_COMPLEMENT_MODE+x} && "$MVS_COMPLEMENT_MODE" != "$complement_mode" ]]; then
    echo "${variant} fixes MVS_COMPLEMENT_MODE=${complement_mode}; unset the conflicting MVS_COMPLEMENT_MODE=${MVS_COMPLEMENT_MODE}." >&2
    exit 2
fi

n_views=${N_VIEWS:-$default_views}
gpu_id=${GPU_ID:-0}
if ! [[ "$n_views" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_VIEWS must be one positive integer." >&2; exit 2
fi
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one non-negative integer." >&2; exit 2
fi
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ ${ITERATIONS+x} ]] && ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ITERATIONS must be a positive integer." >&2; exit 2
fi
if [[ ${RESOLUTION+x} ]] && ! [[ "$RESOLUTION" =~ ^(-1|[1-9][0-9]*)$ ]]; then
    echo "RESOLUTION must be a positive integer or -1 (automatic)." >&2; exit 2
fi

effective_mode=$logical_mode
method_args=()
if [[ "$dataset" == blender ]]; then
    effective_mode=dropgaussian
    echo "${variant}/Blender: original random initialization (10000 points), no MVS; density_control=${density}." >&2
    if [[ "$complement" == 1 ]]; then
        echo "The complement module is inactive on Blender, matching the complete-method dataset policy." >&2
    fi
elif [[ "$complement" == 1 ]]; then
    max_points=${MVS_COMPLEMENT_MAX_POINTS:-100000}
    total_points=${MVS_COMPLEMENT_TOTAL_POINTS:-0}
    candidate_points=${MVS_COMPLEMENT_CANDIDATE_POINTS:-500000}
    coverage_pixels=${MVS_COMPLEMENT_COVERAGE_PIXELS:-2.0}
    depth_scale=${MVS_COMPLEMENT_DEPTH_SCALE:-0.02}
    for name in max_points candidate_points; do
        if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
            echo "$name must be a positive integer." >&2; exit 2
        fi
    done
    if ! [[ "$total_points" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "total_points must be zero or a positive integer." >&2; exit 2
    fi
    for name in coverage_pixels depth_scale; do
        if ! [[ "${!name}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
            ! awk -v value="${!name}" 'BEGIN { number = value + 0; formatted = sprintf("%g", number); exit !(number > 0 && formatted !~ /[Ii][Nn][Ff]|[Nn][Aa][Nn]/) }'; then
            echo "$name must be a positive finite number." >&2; exit 2
        fi
    done
    method_args=(--mvs_complement_mode coverage
        --mvs_complement_max_points "$max_points"
        --mvs_complement_total_points "$total_points"
        --mvs_complement_candidate_points "$candidate_points"
        --mvs_complement_coverage_pixels "$coverage_pixels"
        --mvs_complement_depth_scale "$depth_scale")
fi

output_parent=${OUTPUT_ROOT:-$repo_root/output/ablation}
setting_output="${output_parent%/}/${variant}/${dataset}${n_views}"
if [[ "$dry_run" == 0 && ( -e "$setting_output" || -L "$setting_output" ) ]]; then
    if [[ -L "$setting_output" || ! -d "$setting_output" ]]; then
        echo "Output setting must be a real directory, not a file or symlink: $setting_output" >&2
        exit 2
    fi
    shopt -s nullglob dotglob
    output_entries=("$setting_output"/*)
    if (( ${#output_entries[@]} )); then
        echo "Output setting directory is nonempty: $setting_output" >&2
        echo "Use a fresh OUTPUT_ROOT to avoid mixing different experiments or reusing stale results." >&2
        exit 2
    fi
fi

command=(env "MODE=$effective_mode" "DATASET=$dataset" "DENSITY_CONTROL=$density"
    "N_VIEWS=$n_views" "GPU_ID=$gpu_id" "OUTPUT_ROOT=$setting_output"
    bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh"
    "${train_args[@]}" "${method_args[@]}")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
