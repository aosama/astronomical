#!/usr/bin/env sh

# Reclaims disk space from accumulated stale Cargo target artifacts.
#
# Why this exists: Cargo never deletes an artifact whose output-hash changed, and
# this workspace intentionally exercises many package/feature combinations, so
# every distinct unification leaves its own variant in target/debug/deps forever
# (measured: 363 variants of one test binary, 27.1 GiB across 22,385 files).
# Cargo has no built-in target-directory garbage collection, and age-based
# selective sweeping would contradict the repository rule that per-generation
# ownership must never be inferred from file age. The repository answer is a
# deliberate whole-target `cargo clean`, which this script wraps with two
# safeguards: the acceptance-evidence tree inside target/ is preserved across
# the clean, and deletion is owned entirely by cargo clean so this script never
# runs rm against repository paths.
#
# Cargo's own caches (the sccache store and the native build store under
# ~/Library/Caches/Astronomical/) live outside target/ and are untouched, so
# the following rebuild repopulates target/ from warm caches.

set -eu
# pipefail is Bash/Zsh; the subshell probe keeps this script POSIX-runnable.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    set -o pipefail
fi

readonly EVIDENCE_DIRECTORY_NAME="acceptance-evidence"
readonly STASH_DIRECTORY_PREFIX="astronomical-target-cleanup-stash"

REPOSITORY_ROOT=""
TARGET_DIRECTORY=""
EVIDENCE_DIRECTORY=""
STASH_DIRECTORY=""
SHOULD_APPLY="false"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    print_error "Usage: scripts/clean-cargo-target-artifacts.sh [--apply]"
    print_error "Without --apply the script only reports the reclaimable size."
}

cleanup() {
    # The stash must never outlive this process with evidence inside it: if the
    # clean or the restore failed, put the evidence back before exiting so no
    # acceptance evidence is ever lost by an interrupted run.
    if [ -n "${STASH_DIRECTORY:-}" ] && [ -d "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}" ]; then
        if [ -d "${TARGET_DIRECTORY}" ]; then
            mv "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}" "${TARGET_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}"
        else
            mkdir -p "${TARGET_DIRECTORY}"
            mv "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}" "${TARGET_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}"
        fi
        printf '%s\n' "Restored acceptance evidence after an interrupted run." >&2
    fi
    if [ -n "${STASH_DIRECTORY:-}" ] && [ -d "${STASH_DIRECTORY}" ]; then
        case "${STASH_DIRECTORY}" in
            /|.|..)
                print_error "refusing to remove unsafe cleanup stash directory: ${STASH_DIRECTORY}"
                ;;
            *)
                rmdir "${STASH_DIRECTORY}" 2>/dev/null || true
                ;;
        esac
    fi
}
trap cleanup 0

resolve_repository_root() {
    repository_root_candidate="$(dirname -- "$0")/.."
    REPOSITORY_ROOT="$(CDPATH='' cd -- "${repository_root_candidate}" && pwd -P)" || {
        print_error "repository root is unavailable: ${repository_root_candidate}"
        exit 1
    }
    [ -f "${REPOSITORY_ROOT}/Cargo.toml" ] || {
        print_error "Cargo.toml is missing beside the resolved repository root: ${REPOSITORY_ROOT}"
        exit 1
    }
    TARGET_DIRECTORY="${REPOSITORY_ROOT}/target"
    EVIDENCE_DIRECTORY="${TARGET_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}"
}

# Prints the directory size in kibibytes so callers convert once, in one place,
# to the decimal SI gigabytes the repository mandates for user-facing sizes.
directory_kibibytes() {
    measured_directory="$1"
    if [ -d "${measured_directory}" ]; then
        du -sk "${measured_directory}" | cut -f1
    else
        printf '0\n'
    fi
}

# Converts kibibytes to decimal SI gigabytes (1 GB = 1,000,000,000 bytes).
format_decimal_gigabytes() {
    size_kibibytes="$1"
    awk -v kib="${size_kibibytes}" 'BEGIN { printf "%.2f GB", (kib * 1024) / 1000000000 }'
}

stash_acceptance_evidence() {
    [ -d "${EVIDENCE_DIRECTORY}" ] || return 0
    STASH_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/${STASH_DIRECTORY_PREFIX}.XXXXXX")"
    case "${STASH_DIRECTORY}" in
        /|.|..)
            print_error "refusing to use unsafe cleanup stash directory: ${STASH_DIRECTORY}"
            exit 1
            ;;
    esac
    mv "${EVIDENCE_DIRECTORY}" "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}"
}

restore_acceptance_evidence() {
    [ -n "${STASH_DIRECTORY}" ] || return 0
    [ -d "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}" ] || {
        STASH_DIRECTORY=""
        return 0
    }
    mkdir -p "${TARGET_DIRECTORY}"
    mv "${STASH_DIRECTORY}/${EVIDENCE_DIRECTORY_NAME}" "${EVIDENCE_DIRECTORY}"
    STASH_DIRECTORY=""
}

report_step() {
    step_name="$1"
    step_started_at_seconds="$2"
    printf '[clean-cargo-target-artifacts] step=%s elapsed_seconds=%s\n' \
        "${step_name}" "$(( $(date +%s) - step_started_at_seconds ))"
}

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --apply)
                SHOULD_APPLY="true"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unsupported option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    resolve_repository_root

    run_started_at_seconds="$(date +%s)"
    size_before_kibibytes="$(directory_kibibytes "${TARGET_DIRECTORY}")"
    evidence_kibibytes="$(directory_kibibytes "${EVIDENCE_DIRECTORY}")"
    reclaimable_kibibytes=$(( size_before_kibibytes - evidence_kibibytes ))

    printf 'target directory: %s (%s)\n' "${TARGET_DIRECTORY}" "$(format_decimal_gigabytes "${size_before_kibibytes}")"
    printf 'preserved acceptance evidence: %s\n' "$(format_decimal_gigabytes "${evidence_kibibytes}")"
    printf 'reclaimable stale artifacts: %s\n' "$(format_decimal_gigabytes "${reclaimable_kibibytes}")"

    if [ "${SHOULD_APPLY}" != "true" ]; then
        printf '%s\n' "Dry run only. Re-run with --apply to clean."
        report_step "dry-run" "${run_started_at_seconds}"
        exit 0
    fi

    stash_started_at_seconds="$(date +%s)"
    stash_acceptance_evidence
    report_step "stash-acceptance-evidence" "${stash_started_at_seconds}"

    clean_started_at_seconds="$(date +%s)"
    printf 'Running cargo clean (live output follows)...\n'
    (cd "${REPOSITORY_ROOT}" && cargo clean)
    report_step "cargo-clean" "${clean_started_at_seconds}"

    restore_started_at_seconds="$(date +%s)"
    restore_acceptance_evidence
    report_step "restore-acceptance-evidence" "${restore_started_at_seconds}"

    size_after_kibibytes="$(directory_kibibytes "${TARGET_DIRECTORY}")"
    freed_kibibytes=$(( size_before_kibibytes - size_after_kibibytes ))
    printf 'target directory after clean: %s (%s freed)\n' \
        "$(format_decimal_gigabytes "${size_after_kibibytes}")" "$(format_decimal_gigabytes "${freed_kibibytes}")"
    report_step "total" "${run_started_at_seconds}"
}

main "$@"
