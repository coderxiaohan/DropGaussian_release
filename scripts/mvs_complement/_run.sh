#!/usr/bin/env bash
set -euo pipefail

# Geometry-complement initialization on real datasets, original random
# initialization on Blender. Downstream DropGaussian and VCD/VCP are shared.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
dataset=${1:?A dataset is required}
shift
case "$dataset" in
    llff|dtu) default_views=3 ;;
    mipnerf360) default_views=12 ;;
    blender) default_views=8 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

dry_run=${DRY_RUN:-0}
train_args=()
while (( $# )); do
    argument_name=${1%%=*}
    case "$argument_name" in
        -m*) echo "Use OUTPUT_ROOT, not $argument_name; this wrapper owns model paths." >&2; exit 2 ;;
        -s*) echo "Use DATA_ROOT and SCENES, not $argument_name; this wrapper owns source paths." >&2; exit 2 ;;
        -r*) echo "Use RESOLUTION, not $argument_name; training and evaluation must agree." >&2; exit 2 ;;
        -i*|-w*) echo "This wrapper fixes image-loading and background protocol options." >&2; exit 2 ;;
        --mvs*|--density*|--vcd*|--vcp*)
            echo "Use MVS_COMPLEMENT_* or supported MVS_INIT_* environment settings; method CLI overrides are disabled." >&2
            exit 2
            ;;
        --*)
            # argparse accepts long-option abbreviations; reserve these too.
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
                    echo "This wrapper owns $reserved; use its documented environment settings, not $argument_name." >&2
                    exit 2
                fi
            done
            ;;
    esac
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/mvs_complement/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
                '' \
                'Real datasets: preserve reconstructed geometry and use MVS to complement training-view coverage.' \
                'Blender main method: original DropGaussian random initialization (10000 points), without MVS.' \
                'Only initialization changes; DropGaussian and VCD/VCP (EGD/TBP) remain unchanged.' \
                "Defaults: all configured ${dataset} scenes, ${default_views} training views, 10000 iterations, GPU 0." \
                'All scenes does not mean all input views; N_VIEWS controls sparse-view training.' \
                'MVS_COMPLEMENT_MODE: coverage (default), concat, balanced, or none.' \
                'Blender coverage/concat/balanced all use random initialization, with no MVS cache or depth preflight.' \
                'Blender random initialization ignores all MVS_COMPLEMENT_* budgets/selection settings and MVS_INIT_* settings.' \
                'The coverage/concat/balanced selection and budgets below apply to real datasets only.' \
                'coverage: coverage-aware MVS selection; requires compatible training-view depth caches.' \
                'concat: original reconstructed cloud plus MVS, without coverage selection.' \
                'balanced: same source quotas and depth-supported, positive-novelty candidate pool as coverage,' \
                '          but samples that pool uniformly; also requires compatible training-view depth caches.' \
                'none: explicit direct-MVS ablation on every dataset, including Blender; no original geometry is retained.' \
                'none with TOTAL_POINTS>0 sets the direct-MVS budget to that total (overrides MVS_INIT_MAX_POINTS).' \
                'none with TOTAL_POINTS=0 uses MVS_INIT_MAX_POINTS (default: 100000).' \
                'MVS_COMPLEMENT_MAX_POINTS: added MVS budget (default: 100000).' \
                'MVS_COMPLEMENT_TOTAL_POINTS: 0 preserves all original geometry (default); positive values cap the total.' \
                'A positive total budget may subsample original geometry; it does not preserve every original point.' \
                'Use the same positive total budget for fair point-count comparisons, e.g. 100000.' \
                'MVS_COMPLEMENT_CANDIDATE_POINTS: MVS candidate pool (default: 500000).' \
                'MVS_COMPLEMENT_COVERAGE_PIXELS: coverage radius in depth-cache pixels, not resized training-image pixels (default: 2.0).' \
                'MVS_COMPLEMENT_DEPTH_SCALE: relative-depth agreement scale (default: 0.02).' \
                'Random clouds are not reconstructed geometry; Blender random initialization is an explicit dataset policy.' \
                'Real datasets without valid reconstructed geometry use direct MVS and record the fallback.' \
                'Baseline geometry assets are reused; their training-view-only provenance is not independently verified.' \
                'MVS branches treat missing/incompatible caches as errors, not random-cloud fallbacks.' \
                'Blender random forces MODE=dropgaussian; MVS branches force MODE=mvs and strategy=replace.' \
                'All branches force density_control=vcd_vcp.' \
                'Overrides: N_VIEWS, GPU_ID, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION,' \
                'PYTHON_BIN, OUTPUT_ROOT, MVS_INIT_PATH, MVS_INIT_MAX_POINTS, DTU_MASK_ROOT, SEED, PORT.' \
                'Use environment settings, not CLI source/model/dataset/iteration/resolution/method options.' \
                'OUTPUT_ROOT is a parent: Blender random uses <root>/blender_random/blender<views>.' \
                'MVS branches use <root>/<mode>/<keep_base or budgetN>/<dataset><views>.' \
                'Default parent: <repo>/output/mvs_complement. Existing nonempty settings are never reused.' \
                "Example: GPU_ID=0 N_VIEWS=${default_views} bash scripts/mvs_complement/train_${dataset}.sh" \
                'DRY_RUN=1 or --dry-run prints dispatch only, without data checks or output writes.' \
                'Train -> render -> evaluate -> summarize uses the existing common runner.'
            exit 0
            ;;
        --) echo "The -- option separator is not supported; pass train.py arguments directly." >&2; exit 2 ;;
        *) train_args+=("$1"); shift ;;
    esac
done

complement_mode=${MVS_COMPLEMENT_MODE:-coverage}
case "$complement_mode" in
    none|coverage|concat|balanced) ;;
    *) echo "MVS_COMPLEMENT_MODE must be none, coverage, concat, or balanced." >&2; exit 2 ;;
esac
blender_random=0
if [[ "$dataset" == blender && "$complement_mode" != none ]]; then
    blender_random=1
fi
if [[ "$blender_random" == 0 ]]; then
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
        echo "MVS_COMPLEMENT_TOTAL_POINTS must be zero or a positive integer." >&2; exit 2
    fi
    for name in coverage_pixels depth_scale; do
        if ! [[ "${!name}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
            ! awk -v value="${!name}" 'BEGIN { number = value + 0; formatted = sprintf("%g", number); exit !(number > 0 && formatted !~ /[Ii][Nn][Ff]|[Nn][Aa][Nn]/) }'; then
            echo "$name must be a positive finite number." >&2; exit 2
        fi
    done
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

output_parent=${OUTPUT_ROOT:-$repo_root/output/mvs_complement}
if [[ "$blender_random" == 1 ]]; then
    setting_output="${output_parent%/}/blender_random/blender${n_views}"
else
    budget_label=keep_base
    if [[ "$total_points" != 0 ]]; then
        budget_label="budget${total_points}"
    fi
    setting_output="${output_parent%/}/${complement_mode}/${budget_label}/${dataset}${n_views}"
fi
if [[ "$dry_run" == 0 && ( -e "$setting_output" || -L "$setting_output" ) ]]; then
    if [[ ! -d "$setting_output" ]]; then
        echo "Output setting path already exists and is not a directory: $setting_output" >&2
        exit 2
    fi
    shopt -s nullglob dotglob
    output_entries=("$setting_output"/*)
    if (( ${#output_entries[@]} )); then
        echo "Output setting directory is nonempty: $setting_output" >&2
        echo "Use a fresh OUTPUT_ROOT to avoid reusing results from different initialization settings." >&2
        exit 2
    fi
fi
if [[ "$blender_random" == 1 ]]; then
    echo "Blender main method: original DropGaussian random initialization (10000 points), no MVS/cache requirement; VCD/VCP (EGD/TBP) retained." >&2
    echo "Blender ignores MVS_COMPLEMENT_* budgets/selection settings and MVS_INIT_* settings; use MVS_COMPLEMENT_MODE=none for the direct-MVS ablation." >&2
    command=(env MODE=dropgaussian DENSITY_CONTROL=vcd_vcp
        "N_VIEWS=$n_views" "GPU_ID=$gpu_id" "OUTPUT_ROOT=$setting_output"
        bash "$script_dir/../dropgaussian_vcd_vcp/train_blender.sh" "${train_args[@]}")
else
    command=(env MODE=mvs MVS_INIT_STRATEGY=replace DENSITY_CONTROL=vcd_vcp)
    if [[ "$complement_mode" == none && "$total_points" != 0 ]]; then
        # The core's mode=none bypasses complement processing. Apply the requested
        # fair-comparison budget to the unchanged direct-MVS loader itself.
        command+=("MVS_INIT_MAX_POINTS=$total_points")
    fi
    command+=(
        "N_VIEWS=$n_views" "GPU_ID=$gpu_id" "OUTPUT_ROOT=$setting_output"
        bash "$script_dir/../mvs_vcd_vcp/train_${dataset}.sh"
        "${train_args[@]}"
        --mvs_complement_mode "$complement_mode"
        --mvs_complement_max_points "$max_points"
        --mvs_complement_total_points "$total_points"
        --mvs_complement_candidate_points "$candidate_points"
        --mvs_complement_coverage_pixels "$coverage_pixels"
        --mvs_complement_depth_scale "$depth_scale")
fi
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
