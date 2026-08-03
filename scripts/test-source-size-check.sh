#!/usr/bin/env sh

set -eu

readonly MAXIMUM_ELAPSED_SECONDS=120
SCRIPT_DIRECTORY="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
SOURCE_SIZE_CHECKER="${SCRIPT_DIRECTORY}/check-source-size.sh"
PRE_COMMIT_HOOK="${SCRIPT_DIRECTORY}/../.githooks/pre-commit"
TEMPORARY_DIRECTORY=""

cleanup() {
    case "${TEMPORARY_DIRECTORY:-}" in
        ""|/|.|..)
            return
            ;;
    esac
    if [ -d "${TEMPORARY_DIRECTORY}" ]; then
        rm -rf "${TEMPORARY_DIRECTORY}"
    fi
}
trap cleanup 0

write_rust_source_file() {
    source_file="$1"
    source_line_count="$2"
    source_header="${3:-}"
    : > "${source_file}"
    if [ -n "${source_header}" ]; then
        printf '%s\n' "${source_header}" >> "${source_file}"
        source_line_count=$((source_line_count - 1))
    fi
    while [ "${source_line_count}" -gt 0 ]; do
        printf '%s\n' '// structural source-size test line' >> "${source_file}"
        source_line_count=$((source_line_count - 1))
    done
}

write_rust_source_file_without_final_newline() {
    source_file="$1"
    source_line_count="$2"
    : > "${source_file}"
    while [ "${source_line_count}" -gt 1 ]; do
        printf '%s\n' '// structural source-size test line' >> "${source_file}"
        source_line_count=$((source_line_count - 1))
    done
    printf '%s' '// structural source-size final line' >> "${source_file}"
}

write_source_file() {
    source_file="$1"
    source_line_count="$2"
    source_line_prefix="$3"
    source_header="${4:-}"
    : > "${source_file}"
    if [ -n "${source_header}" ]; then
        printf '%s\n' "${source_header}" >> "${source_file}"
        source_line_count=$((source_line_count - 1))
    fi
    while [ "${source_line_count}" -gt 0 ]; do
        printf '%s\n' "${source_line_prefix}" >> "${source_file}"
        source_line_count=$((source_line_count - 1))
    done
}

initialize_staged_hook_repository() {
    staged_hook_repository_root="$1"
    mkdir -p "${staged_hook_repository_root}/scripts" "${staged_hook_repository_root}/.githooks"
    git -C "${staged_hook_repository_root}" init --quiet
    git -C "${staged_hook_repository_root}" config user.email 'source-size-test@example.invalid'
    git -C "${staged_hook_repository_root}" config user.name 'Source Size Test'
    git -C "${staged_hook_repository_root}" config core.hooksPath .githooks
    cp "${SOURCE_SIZE_CHECKER}" "${staged_hook_repository_root}/scripts/check-source-size.sh"
    cp "${PRE_COMMIT_HOOK}" "${staged_hook_repository_root}/.githooks/pre-commit"
    chmod +x \
        "${staged_hook_repository_root}/scripts/check-source-size.sh" \
        "${staged_hook_repository_root}/.githooks/pre-commit"
}

run_staged_pre_commit_hook() {
    staged_hook_repository_root="$1"
    (
        cd "${staged_hook_repository_root}"
        ./.githooks/pre-commit
    )
}

main() {
    started_epoch_seconds="$(date +%s)"
    printf '%s\n' "[source-size-test] status=start scenarios=11 timeout_seconds=${MAXIMUM_ELAPSED_SECONDS} ETA_seconds=10"
    TEMPORARY_DIRECTORY="$(mktemp -d)"
    warning_root="${TEMPORARY_DIRECTORY}/warning"
    failing_root="${TEMPORARY_DIRECTORY}/failing"
    generated_root="${TEMPORARY_DIRECTORY}/generated"
    misleading_generated_marker_root="${TEMPORARY_DIRECTORY}/misleading-generated-marker"
    missing_final_newline_root="${TEMPORARY_DIRECTORY}/missing-final-newline"
    staged_hook_repository_root="${TEMPORARY_DIRECTORY}/staged-hook-repository"
    mkdir -p \
        "${warning_root}/src" \
        "${failing_root}/src" \
        "${generated_root}/src" \
        "${misleading_generated_marker_root}/src" \
        "${missing_final_newline_root}/src"
    write_rust_source_file "${warning_root}/src/warning.rs" 501
    write_rust_source_file "${failing_root}/src/failing.rs" 551
    write_rust_source_file "${generated_root}/src/generated.rs" 600 '// @generated deterministic binding'
    write_rust_source_file \
        "${misleading_generated_marker_root}/src/hand_written.rs" \
        600 \
        '// This hand-written source mentions @generated but is not generated.'
    write_rust_source_file_without_final_newline \
        "${missing_final_newline_root}/src/failing_without_final_newline.rs" \
        551

    printf '%s\n' "[source-size-test] status=progress scenarios=0/11 scenario=warning expected=success ETA_seconds=10"
    warning_scenario_output="$(sh "${SOURCE_SIZE_CHECKER}" "${warning_root}")"
    printf '%s\n' "${warning_scenario_output}"
    case "${warning_scenario_output}" in
        *"[source-size] status=warning lines=501 "*) ;;
        *)
            printf '%s\n' '[source-size-test] status=failed reason=501-lines-did-not-emit-warning' >&2
            exit 1
            ;;
    esac
    printf '%s\n' "[source-size-test] status=progress scenarios=1/11 scenario=generated expected=success ETA_seconds=9"
    generated_scenario_output="$(sh "${SOURCE_SIZE_CHECKER}" "${generated_root}")"
    printf '%s\n' "${generated_scenario_output}"
    case "${generated_scenario_output}" in
        *"[source-size] status=warning "*|*"[source-size] status=failed lines="*)
            printf '%s\n' '[source-size-test] status=failed reason=explicit-generated-binding-was-checked' >&2
            exit 1
            ;;
    esac
    printf '%s\n' "[source-size-test] status=progress scenarios=2/11 scenario=misleading-generated-marker expected=failure ETA_seconds=8"
    if sh "${SOURCE_SIZE_CHECKER}" "${misleading_generated_marker_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=misleading-generated-marker-was-excluded' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=3/11 scenario=failing expected=failure ETA_seconds=7"
    if sh "${SOURCE_SIZE_CHECKER}" "${failing_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=551-lines-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=4/11 scenario=missing-final-newline expected=failure ETA_seconds=6"
    if sh "${SOURCE_SIZE_CHECKER}" "${missing_final_newline_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=551-lines-without-final-newline-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=5/11 scenario=staged-rust-boundary expected=success ETA_seconds=5"
    initialize_staged_hook_repository "${staged_hook_repository_root}"
    write_source_file "${staged_hook_repository_root}/boundary.rs" 600 '// staged Rust source line'
    git -C "${staged_hook_repository_root}" add boundary.rs
    run_staged_pre_commit_hook "${staged_hook_repository_root}"
    printf '%s\n' '// unstaged source line beyond the staged boundary' >> "${staged_hook_repository_root}/boundary.rs"
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=6/11 scenario=staged-python expected=success ETA_seconds=4"
    write_source_file "${staged_hook_repository_root}/boundary.py" 600 '# staged Python source line'
    git -C "${staged_hook_repository_root}" add boundary.py
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=7/11 scenario=staged-swift expected=success ETA_seconds=3"
    write_source_file "${staged_hook_repository_root}/Boundary.swift" 600 '// staged Swift source line'
    git -C "${staged_hook_repository_root}" add Boundary.swift
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=8/11 scenario=staged-generated expected=success ETA_seconds=2"
    write_source_file "${staged_hook_repository_root}/generated.py" 601 '# generated Python source line' '# @generated deterministic binding'
    git -C "${staged_hook_repository_root}" add generated.py
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=9/11 scenario=staged-markdown expected=success ETA_seconds=1"
    write_source_file "${staged_hook_repository_root}/oversized.md" 601 'non-code documentation line'
    git -C "${staged_hook_repository_root}" add oversized.md
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=10/11 scenario=staged-limit expected=failure ETA_seconds=0"
    write_source_file "${staged_hook_repository_root}/too_long.cpp" 601 '// staged C++ source line'
    git -C "${staged_hook_repository_root}" add too_long.cpp
    if staged_hook_output="$(run_staged_pre_commit_hook "${staged_hook_repository_root}" 2>&1)"; then
        printf '%s\n' '[source-size-test] status=failed reason=601-line-staged-source-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "${staged_hook_output}"
    case "${staged_hook_output}" in
        *"limit=600 path=too_long.cpp"*) ;;
        *)
            printf '%s\n' '[source-size-test] status=failed reason=staged-limit-failure-was-not-reported' >&2
            exit 1
            ;;
    esac
    if git -C "${staged_hook_repository_root}" commit --quiet -m 'blocked staged source size test'; then
        printf '%s\n' '[source-size-test] status=failed reason=git-commit-accepted-oversized-staged-source' >&2
        exit 1
    fi
    if git -C "${staged_hook_repository_root}" rev-parse --verify HEAD >/dev/null 2>&1; then
        printf '%s\n' '[source-size-test] status=failed reason=blocked-commit-created-history' >&2
        exit 1
    fi
    elapsed_seconds=$(( $(date +%s) - started_epoch_seconds ))
    if [ "${elapsed_seconds}" -gt "${MAXIMUM_ELAPSED_SECONDS}" ]; then
        printf '%s\n' "[source-size-test] status=failed reason=timeout elapsed_seconds=${elapsed_seconds}" >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=success elapsed_seconds=${elapsed_seconds}"
}

main
