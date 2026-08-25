#!/usr/bin/env sh

# Runs large ignored qualification suites one test at a time. Each model or GPU test receives
# the repository's complete 120-second execution boundary instead of sharing one suite-wide timer.

set -eu
# The guarded extension preserves the bounded runner's failure through the live `tee` pipeline.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    # shellcheck disable=SC3040
    set -o pipefail
fi

TEMPORARY_DIRECTORY=""

cleanup() {
    if [ -z "${TEMPORARY_DIRECTORY:-}" ]; then
        return
    fi
    case "${TEMPORARY_DIRECTORY}" in
        /|.|..)
            printf '%s\n' "Error: refusing to remove unsafe temporary directory: ${TEMPORARY_DIRECTORY}" >&2
            return
            ;;
    esac
    if [ -d "${TEMPORARY_DIRECTORY}" ]; then
        rm -rf "${TEMPORARY_DIRECTORY}"
    fi
}
trap cleanup 0

qualification_skip_reason() {
    qualification_test_name="$1"
    case "$qualification_test_name" in
        *persistent_prompt_cache_qualification::cache_interaction_matrix::should_qualify_selected_pinned_ornith_cache_interaction_matrix_cell)
            printf '%s\n' "requires-explicit-selected-cache-cell"
            ;;
        *)
            return 1
            ;;
    esac
}

run_selected_suite() {
    selected_suite="$1"
    case "$selected_suite" in
        model-artifacts)
            set -- --no-fail-fast -p astronomical-model-serving -p astronomical-inference-worker --test model_artifact_qualification_tests --features astronomical-model-serving/direct-mlx,astronomical-inference-worker/model-artifact-qualification
            ;;
        persistent-prompt-cache)
            set -- --no-fail-fast -p astronomical-model-serving -p astronomical-inference-worker --test persistent_prompt_cache_qualification_tests --features astronomical-model-serving/direct-mlx,astronomical-inference-worker/model-artifact-qualification
            ;;
        memory-management)
            set -- -p astronomical-inference-worker --test memory_management_acceptance_tests --features memory-management-acceptance
            ;;
        *)
            printf '%s\n' "Error: unknown ignored qualification suite: ${selected_suite}" >&2
            exit 2
            ;;
    esac

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    bounded_test_runner="${repository_root}/scripts/run-bounded-cargo-test.sh"
    list_output_path="${TEMPORARY_DIRECTORY}/test-list-output"
    test_names_path="${TEMPORARY_DIRECTORY}/test-names"
    list_started_at_seconds="$(date +%s)"
    printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} phase=list status=start started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    if "$bounded_test_runner" cargo test "$@" -- --ignored --list 2>&1 | tee "$list_output_path"; then
        :
    else
        list_exit_status="$?"
        printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} phase=list status=failed elapsed_seconds=$(( $(date +%s) - list_started_at_seconds ))"
        exit "$list_exit_status"
    fi
    : > "$test_names_path"
    while IFS= read -r listed_test_line; do
        case "$listed_test_line" in
            *": test")
                printf '%s\n' "${listed_test_line%: test}" >> "$test_names_path"
                ;;
        esac
    done < "$list_output_path"

    qualification_test_count=0
    while IFS= read -r qualification_test_name; do
        if [ -n "$qualification_test_name" ]; then
            qualification_test_count=$((qualification_test_count + 1))
        fi
    done < "$test_names_path"
    if [ "$qualification_test_count" -eq 0 ]; then
        printf '%s\n' "Error: suite ${selected_suite} did not list any ignored qualification tests" >&2
        exit 1
    fi
    printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} phase=list status=success tests=${qualification_test_count} elapsed_seconds=$(( $(date +%s) - list_started_at_seconds ))"

    completed_test_count=0
    failed_test_count=0
    skipped_test_count=0
    while IFS= read -r qualification_test_name; do
        if [ -z "$qualification_test_name" ]; then
            continue
        fi
        completed_test_count=$((completed_test_count + 1))
        if qualification_skip_reason="$(qualification_skip_reason "$qualification_test_name")"; then
            skipped_test_count=$((skipped_test_count + 1))
            printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} test=${completed_test_count}/${qualification_test_count} status=skipped reason=${qualification_skip_reason} name=${qualification_test_name}"
            continue
        fi
        test_started_at_seconds="$(date +%s)"
        printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} test=${completed_test_count}/${qualification_test_count} status=start name=${qualification_test_name} started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
        if "$bounded_test_runner" cargo test "$@" "$qualification_test_name" -- --ignored --nocapture --exact --test-threads 1; then
            test_status="success"
        else
            test_status="failed"
            failed_test_count=$((failed_test_count + 1))
        fi
        printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} test=${completed_test_count}/${qualification_test_count} status=${test_status} elapsed_seconds=$(( $(date +%s) - test_started_at_seconds )) name=${qualification_test_name}"
    done < "$test_names_path"

    printf '%s\n' "[ignored-qualification-suite] suite=${selected_suite} status=complete tests=${qualification_test_count} failures=${failed_test_count} skipped=${skipped_test_count}"
    if [ "$failed_test_count" -ne 0 ]; then
        exit 1
    fi
}

main() {
    if [ "$#" -ne 1 ]; then
        printf '%s\n' "Usage: scripts/run-ignored-qualification-suite.sh SUITE" >&2
        exit 2
    fi
    TEMPORARY_DIRECTORY="$(mktemp -d)" || {
        printf '%s\n' "Error: failed to create qualification temporary directory" >&2
        exit 1
    }
    run_selected_suite "$1"
}

main "$@"
