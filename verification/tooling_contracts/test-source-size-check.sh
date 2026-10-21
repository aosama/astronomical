#!/usr/bin/env sh

set -eu

readonly MAXIMUM_ELAPSED_SECONDS=120
REPOSITORY_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)"
XTASK_MANIFEST="${REPOSITORY_ROOT}/xtask/Cargo.toml"
TEMPORARY_DIRECTORY=""

# The xtask is built from this checkout's manifest so the scenarios exercise
# the real tool; the staged scenarios run it inside sandbox repositories,
# which is exactly the hook's calling convention.
run_source_size() {
    cargo run --quiet --manifest-path "${XTASK_MANIFEST}" -p xtask -- source-size "$@"
}

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
    mkdir -p "${staged_hook_repository_root}/.githooks"
    git -C "${staged_hook_repository_root}" init --quiet
    # The sandbox has no toolchain file of its own, so rustup would fall back
    # to the machine's default toolchain when the hook runs cargo from inside
    # it. Copying the checkout's pin keeps the sandbox on the same compiler.
    cp "${REPOSITORY_ROOT}/rust-toolchain.toml" "${staged_hook_repository_root}/rust-toolchain.toml"
    git -C "${staged_hook_repository_root}" config user.email 'source-size-test@example.invalid'
    git -C "${staged_hook_repository_root}" config user.name 'Source Size Test'
    git -C "${staged_hook_repository_root}" config core.hooksPath .githooks
    # The sandbox hook mirrors the real pre-commit hook's calling convention:
    # the xtask builds from the real checkout's manifest while the staged scan
    # follows this sandbox repository.
    cat > "${staged_hook_repository_root}/.githooks/pre-commit" <<WRAPPER
#!/usr/bin/env sh
set -eu
cargo run --quiet --manifest-path "${XTASK_MANIFEST}" -p xtask -- source-size --staged
WRAPPER
    chmod +x "${staged_hook_repository_root}/.githooks/pre-commit"
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
    printf '%s\n' "[source-size-test] status=start scenarios=12 timeout_seconds=${MAXIMUM_ELAPSED_SECONDS} ETA_seconds=10"
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
    write_rust_source_file "${warning_root}/src/warning.rs" 601
    write_rust_source_file "${failing_root}/src/failing.rs" 651
    write_rust_source_file "${generated_root}/src/generated.rs" 600 '// @generated deterministic binding'
    write_rust_source_file \
        "${misleading_generated_marker_root}/src/hand_written.rs" \
        651 \
        '// This hand-written source mentions @generated but is not generated.'
    write_rust_source_file_without_final_newline \
        "${missing_final_newline_root}/src/failing_without_final_newline.rs" \
        651

    printf '%s\n' "[source-size-test] status=progress scenarios=0/12 scenario=warning expected=success ETA_seconds=10"
    warning_scenario_output="$(run_source_size "${warning_root}")"
    printf '%s\n' "${warning_scenario_output}"
    case "${warning_scenario_output}" in
        *"[source-size] status=warning lines=601 "*) ;;
        *)
            printf '%s\n' '[source-size-test] status=failed reason=601-lines-did-not-emit-warning' >&2
            exit 1
            ;;
    esac
    printf '%s\n' "[source-size-test] status=progress scenarios=1/12 scenario=generated expected=success ETA_seconds=9"
    generated_scenario_output="$(run_source_size "${generated_root}")"
    printf '%s\n' "${generated_scenario_output}"
    case "${generated_scenario_output}" in
        *"[source-size] status=warning "*|*"[source-size] status=failed lines="*)
            printf '%s\n' '[source-size-test] status=failed reason=explicit-generated-binding-was-checked' >&2
            exit 1
            ;;
    esac
    printf '%s\n' "[source-size-test] status=progress scenarios=2/12 scenario=misleading-generated-marker expected=failure ETA_seconds=8"
    if run_source_size "${misleading_generated_marker_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=misleading-generated-marker-was-excluded' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=3/12 scenario=failing expected=failure ETA_seconds=7"
    if run_source_size "${failing_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=651-lines-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=4/12 scenario=missing-final-newline expected=failure ETA_seconds=6"
    if run_source_size "${missing_final_newline_root}"; then
        printf '%s\n' '[source-size-test] status=failed reason=651-lines-without-final-newline-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=progress scenarios=5/12 scenario=staged-rust-boundary expected=success ETA_seconds=5"
    initialize_staged_hook_repository "${staged_hook_repository_root}"
    write_source_file "${staged_hook_repository_root}/boundary.rs" 600 '// staged Rust source line'
    git -C "${staged_hook_repository_root}" add boundary.rs
    run_staged_pre_commit_hook "${staged_hook_repository_root}"
    printf '%s\n' '// unstaged source line beyond the staged boundary' >> "${staged_hook_repository_root}/boundary.rs"
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=6/12 scenario=staged-python expected=success ETA_seconds=4"
    write_source_file "${staged_hook_repository_root}/boundary.py" 600 '# staged Python source line'
    git -C "${staged_hook_repository_root}" add boundary.py
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=7/12 scenario=staged-swift expected=success ETA_seconds=3"
    write_source_file "${staged_hook_repository_root}/Boundary.swift" 600 '// staged Swift source line'
    git -C "${staged_hook_repository_root}" add Boundary.swift
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=8/12 scenario=staged-generated expected=success ETA_seconds=2"
    write_source_file "${staged_hook_repository_root}/generated.py" 601 '# generated Python source line' '# @generated deterministic binding'
    git -C "${staged_hook_repository_root}" add generated.py
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=9/12 scenario=staged-markdown expected=success ETA_seconds=1"
    write_source_file "${staged_hook_repository_root}/oversized.md" 601 'non-code documentation line'
    git -C "${staged_hook_repository_root}" add oversized.md
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    printf '%s\n' "[source-size-test] status=progress scenarios=10/12 scenario=staged-limit expected=failure ETA_seconds=0"
    write_source_file "${staged_hook_repository_root}/too_long.cpp" 701 '// staged C++ source line'
    git -C "${staged_hook_repository_root}" add too_long.cpp
    if staged_hook_output="$(run_staged_pre_commit_hook "${staged_hook_repository_root}" 2>&1)"; then
        printf '%s\n' '[source-size-test] status=failed reason=701-line-staged-source-was-accepted' >&2
        exit 1
    fi
    printf '%s\n' "${staged_hook_output}"
    case "${staged_hook_output}" in
        *"limit=700 path=too_long.cpp"*) ;;
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
    printf '%s\n' "[source-size-test] status=progress scenarios=11/12 scenario=staged-vendored-asset expected=success ETA_seconds=0"
    # Scenario 10 leaves an oversized file staged, so this case stages only the
    # vendored asset to prove the exemption rather than the earlier failure.
    git -C "${staged_hook_repository_root}" reset --quiet
    mkdir -p "${staged_hook_repository_root}/Sources/App/Resources/web/vendor"
    write_source_file \
        "${staged_hook_repository_root}/Sources/App/Resources/web/vendor/bundle.min.js" \
        900 \
        'var vendored=1;'
    git -C "${staged_hook_repository_root}" add \
        Sources/App/Resources/web/vendor/bundle.min.js
    run_staged_pre_commit_hook "${staged_hook_repository_root}"

    elapsed_seconds=$(( $(date +%s) - started_epoch_seconds ))
    if [ "${elapsed_seconds}" -gt "${MAXIMUM_ELAPSED_SECONDS}" ]; then
        printf '%s\n' "[source-size-test] status=failed reason=timeout elapsed_seconds=${elapsed_seconds}" >&2
        exit 1
    fi
    printf '%s\n' "[source-size-test] status=success elapsed_seconds=${elapsed_seconds}"
}

main
