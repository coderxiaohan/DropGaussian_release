#!/usr/bin/env bash
set -euo pipefail

# Same geometry-complementary initializer and EGD as the full method, with an
# opt-in FastGS count-based pruning policy. Existing scripts/results stay intact.
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
    case "$1" in
        --dry-run) dry_run=1; shift; continue ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/fastgs_pruning/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
                '' \
                'Full geometry-complement + unchanged EGD, with FastGS count-based pruning.' \
                'FASTGS_PRUNE_INTERVAL=8000: refresh counts after ADC at matching training iterations.' \
                'FASTGS_LOSS_THRESHOLD=0.1, FASTGS_PRUNE_THRESHOLD=0.9, FASTGS_OPACITY_THRESHOLD=0.1.' \
                'FASTGS_PROTECT_EGD=0 (default) preserves the unprotected FastGS comparison; explicit 1 enables same-event EGD protection.' \
                'Direct threshold-based deletion; no trial-removal validation or subset backoff.' \
                'This is an unverified experiment, not a claim of better quality or speed.' \
                'Real datasets: coverage-based MVS complement; Blender: original random 10000-point initialization.' \
                'EGD, DropGaussian regularization and evaluation remain unchanged; only pruning policy changes.' \
                "Defaults: all configured ${dataset} scenes, ${default_views} views, 10000 iterations, GPU 0." \
                'Overrides: N_VIEWS, GPU_ID, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION,' \
                'PYTHON_BIN, OUTPUT_ROOT, DTU_MASK_ROOT, SEED, PORT and the four FASTGS_* settings above.' \
                'Real datasets also support MVS_INIT_PATH, MVS_INIT_MAX_POINTS and numeric MVS_COMPLEMENT_* settings.' \
                'MVS_COMPLEMENT_MODE=coverage and DENSITY_CONTROL=vcd_vcp are fixed; conflicting settings are rejected.' \
                'Use FASTGS_* environment settings, not --fastgs_* or --vcp_strategy; method CLI overrides are disabled.' \
                'Scoring uses every distinct training view; upstream FastGS samples three views.' \
                'This is a pruning-policy adaptation, not a reproduction of the full FastGS training pipeline.' \
                'FASTGS_PRUNE_INTERVAL must be smaller than ITERATIONS; no final-step pruning is scheduled.' \
                'Training rejects pruning schedules that overlap opacity resets during densification.' \
                'Training startup prepares the isolated CUDA count backend before loading the scene; a CUDA toolkit is required.' \
                'OUTPUT_ROOT is a parent, default <repo>/output/fastgs_pruning.' \
                'Real output: <root>/coverage/<keep_base or budgetN>/<dataset><views>/<scene>.' \
                'Blender output: <root>/blender_random/blender<views>/<scene>.' \
                'Existing nonempty setting directories are rejected. Change OUTPUT_ROOT to repeat a setting.' \
                '--dry-run or DRY_RUN=1 prints dispatch only; no data, GPU, directory writes or training.'
            exit 0
            ;;
        --) echo "The -- option separator is not supported." >&2; exit 2 ;;
    esac
    argument_name=${1%%=*}
    case "$argument_name" in
        -m*) echo "Use OUTPUT_ROOT, not $argument_name." >&2; exit 2 ;;
        -s*) echo "Use DATA_ROOT and SCENES, not $argument_name." >&2; exit 2 ;;
        -r*) echo "Use RESOLUTION, not $argument_name." >&2; exit 2 ;;
        -i*|-w*) echo "Image and background protocol options are fixed." >&2; exit 2 ;;
        --mvs*|--density*|--vcd*|--vcp*|--fastgs*|--dpp*)
            echo "Method CLI overrides are disabled; use documented FASTGS_* and MVS_* numeric environment settings." >&2
            exit 2
            ;;
        --*)
            # Reserve argparse abbreviations as well as complete options.
            for reserved in --model_path --source_path --iterations --n_views \
                --dataset_name --resolution --mvs_complement_mode --density_control \
                --init_mode --mvs_init_strategy --rand_pcd --white_background \
                --images --eval --llffhold --seed --test_iterations --save_iterations \
                --start_checkpoint --sh_degree --position_lr_init --position_lr_final \
                --position_lr_delay_mult --position_lr_max_steps --feature_lr \
                --opacity_lr --scaling_lr --rotation_lr --percent_dense --lambda_dssim \
                --densification_interval --opacity_reset_interval --densify_from_iter \
                --densify_until_iter --densify_grad_threshold --random_background \
                --vcd_error_quantile --vcp_strategy \
                --fastgs_prune_interval --fastgs_loss_threshold --fastgs_prune_threshold \
                --fastgs_opacity_threshold --fastgs_protect_egd --dpp_only; do
                if [[ "$reserved" == "$argument_name"* ]]; then
                    echo "This launcher owns $reserved; use documented environment settings." >&2
                    exit 2
                fi
            done
            ;;
    esac
    train_args+=("$1")
    shift
done

prune_interval=${FASTGS_PRUNE_INTERVAL:-8000}
loss_threshold=${FASTGS_LOSS_THRESHOLD:-0.1}
prune_threshold=${FASTGS_PRUNE_THRESHOLD:-0.9}
opacity_threshold=${FASTGS_OPACITY_THRESHOLD:-0.1}
protect_egd=${FASTGS_PROTECT_EGD:-0}
case "$protect_egd" in
    0|1) ;;
    *) echo "FASTGS_PROTECT_EGD must be 0 or 1." >&2; exit 2 ;;
esac
if ! [[ "$prune_interval" =~ ^[1-9][0-9]*$ ]]; then
    echo "FASTGS_PRUNE_INTERVAL must be a positive integer." >&2; exit 2
fi
for name in loss_threshold prune_threshold opacity_threshold; do
    if ! [[ "${!name}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
        ! awk -v value="${!name}" 'BEGIN { number = value + 0; formatted = sprintf("%g", number); exit !(number >= 0 && number <= 1 && formatted !~ /[Ii][Nn][Ff]|[Nn][Aa][Nn]/) }'; then
        echo "$name must be a finite number in [0, 1]." >&2; exit 2
    fi
done
if [[ ${VCP_STRATEGY+x} && "$VCP_STRATEGY" != fastgs ]]; then
    echo "This experiment fixes VCP_STRATEGY=fastgs; unset the conflicting value." >&2
    exit 2
fi
if [[ ${TBP_MAX_TRIALS+x} ]]; then
    echo "TBP_MAX_TRIALS is no longer supported; unset this obsolete setting." >&2
    exit 2
fi
if [[ ${DENSITY_CONTROL+x} && "$DENSITY_CONTROL" != vcd_vcp ]]; then
    echo "This experiment fixes DENSITY_CONTROL=vcd_vcp; unset the conflicting value." >&2
    exit 2
fi
if [[ ${MVS_COMPLEMENT_MODE+x} && "$MVS_COMPLEMENT_MODE" != coverage ]]; then
    echo "This experiment fixes MVS_COMPLEMENT_MODE=coverage; unset the conflicting value." >&2
    exit 2
fi
effective_mode=mvs
if [[ "$dataset" == blender ]]; then
    effective_mode=dropgaussian
fi
if [[ ${MODE+x} && "$MODE" != "$effective_mode" ]]; then
    echo "This dataset fixes MODE=${effective_mode}; unset the conflicting value." >&2
    exit 2
fi
if [[ "$dataset" != blender && ${MVS_INIT_STRATEGY+x} && "$MVS_INIT_STRATEGY" != replace ]]; then
    echo "This experiment fixes MVS_INIT_STRATEGY=replace; unset the conflicting value." >&2
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
if ! awk -v interval="$prune_interval" -v iterations="${ITERATIONS:-10000}" \
    'BEGIN { exit !(interval + 0 < iterations + 0) }'; then
    echo "FASTGS_PRUNE_INTERVAL must be smaller than ITERATIONS (default 10000), leaving training after pruning." >&2
    exit 2
fi
if [[ ${RESOLUTION+x} ]] && ! [[ "$RESOLUTION" =~ ^(-1|[1-9][0-9]*)$ ]]; then
    echo "RESOLUTION must be a positive integer or -1." >&2; exit 2
fi

output_parent=${OUTPUT_ROOT:-$repo_root/output/fastgs_pruning}
output_parent="${output_parent%/}"
method_args=(--vcp_strategy fastgs
    --fastgs_prune_interval "$prune_interval"
    --fastgs_loss_threshold "$loss_threshold"
    --fastgs_prune_threshold "$prune_threshold"
    --fastgs_opacity_threshold "$opacity_threshold")
if [[ "$protect_egd" == 1 ]]; then
    method_args+=(--fastgs_protect_egd)
fi
if [[ "$dataset" == blender ]]; then
    setting_output="$output_parent/blender_random/blender${n_views}"
    echo "Blender: original random initialization (10000 points), no MVS; unchanged EGD + FastGS pruning." >&2
else
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
    budget_label=keep_base
    if [[ "$total_points" != 0 ]]; then
        budget_label="budget${total_points}"
    fi
    setting_output="$output_parent/coverage/$budget_label/${dataset}${n_views}"
    method_args+=(--mvs_complement_mode coverage
        --mvs_complement_max_points "$max_points"
        --mvs_complement_total_points "$total_points"
        --mvs_complement_candidate_points "$candidate_points"
        --mvs_complement_coverage_pixels "$coverage_pixels"
        --mvs_complement_depth_scale "$depth_scale")
fi

if [[ "$dry_run" == 0 && ( -e "$setting_output" || -L "$setting_output" ) ]]; then
    if [[ -L "$setting_output" || ! -d "$setting_output" ]]; then
        echo "Output setting must be a real directory, not a file or symlink: $setting_output" >&2
        exit 2
    fi
    shopt -s nullglob dotglob
    output_entries=("$setting_output"/*)
    if (( ${#output_entries[@]} )); then
        echo "Output setting directory is nonempty: $setting_output" >&2
        echo "Use a fresh OUTPUT_ROOT; existing runs are never reused or overwritten." >&2
        exit 2
    fi
fi

# The complete-method wrapper reserves pruning CLI options. Reuse its exact
# lower dispatch paths without weakening its protection or changing old scripts.
command=(env "MODE=$effective_mode" DENSITY_CONTROL=vcd_vcp MVS_COMPLEMENT_MODE=coverage VCP_STRATEGY=fastgs
    "N_VIEWS=$n_views" "GPU_ID=$gpu_id" "OUTPUT_ROOT=$setting_output")
if [[ "$dataset" == blender ]]; then
    command+=(bash "$script_dir/../dropgaussian_vcd_vcp/train_blender.sh")
else
    command+=(MVS_INIT_STRATEGY=replace bash "$script_dir/../mvs_vcd_vcp/train_${dataset}.sh")
fi
command+=("${train_args[@]}" "${method_args[@]}")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
