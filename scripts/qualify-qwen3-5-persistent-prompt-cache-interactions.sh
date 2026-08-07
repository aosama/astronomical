#!/usr/bin/env sh

set -eu

if [ "$#" -ne 0 ]; then
    printf '%s\n' "Error: this qualification script does not accept arguments" >&2
    exit 2
fi

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    printf '%s\n' "Error: GNU timeout is required; install Homebrew coreutils" >&2
    exit 1
fi

readonly qualification_test_name="persistent_prompt_cache_qualification::cache_interaction_matrix::should_qualify_selected_pinned_ornith_cache_interaction_matrix_cell"
readonly qualification_cells="
fixed-live-reuse
optimized-live-reuse
fixed-worker-restart
optimized-worker-restart
fixed-deleted-while-live
optimized-deleted-while-live
"

completed_cell_count=0
total_cell_count=6
matrix_started_at_seconds="$(date +%s)"

for qualification_cell in ${qualification_cells}; do
    cell_started_at_seconds="$(date +%s)"
    printf '%s\n' "[prompt-cache-interaction-matrix] status=start cell=${qualification_cell} completed=${completed_cell_count}/${total_cell_count} timeout_seconds=120"
    ASTRONOMICAL_PROMPT_CACHE_INTERACTION_QUALIFICATION_CELL="${qualification_cell}" \
        CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}" \
        "${timeout_executable}" -k 5s 120s \
        cargo --verbose test \
            --package astronomical-model-serving \
            --features direct-mlx \
            --test qwen3_5_persistent_prompt_cache_qualification_tests \
            "${qualification_test_name}" \
            -- \
            --ignored \
            --exact \
            --nocapture \
            --test-threads=1
    completed_cell_count=$((completed_cell_count + 1))
    cell_finished_at_seconds="$(date +%s)"
    cell_elapsed_seconds=$((cell_finished_at_seconds - cell_started_at_seconds))
    printf '%s\n' "[prompt-cache-interaction-matrix] status=success cell=${qualification_cell} completed=${completed_cell_count}/${total_cell_count} elapsed_seconds=${cell_elapsed_seconds}"
done

matrix_finished_at_seconds="$(date +%s)"
matrix_elapsed_seconds=$((matrix_finished_at_seconds - matrix_started_at_seconds))
printf '%s\n' "[prompt-cache-interaction-matrix] status=success completed=${completed_cell_count}/${total_cell_count} elapsed_seconds=${matrix_elapsed_seconds}"
