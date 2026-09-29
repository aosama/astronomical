#!/usr/bin/env sh

# Builds Astronomical's pinned native runtime ahead of Cargo so the native
# CMake compile never overlaps Rust compilation. CI, verify-before-commit, and
# the macOS app build all call this before their first Cargo step; the later
# build scripts then find the native build store warm and skip CMake entirely.
#
# The native build tool streams progress into a file (Cargo captures build
# script stderr), so this wrapper tails that file to keep live output visible.

set -eu

readonly SCRIPT_PREFIX="[prewarm-native-build]"

profile_name=""
repository_root=""
progress_tail_pid=""
progress_file=""

cleanup() {
    if [ -n "${progress_tail_pid}" ]; then
        kill "${progress_tail_pid}" 2>/dev/null || true
        wait "${progress_tail_pid}" 2>/dev/null || true
    fi
    if [ -n "${progress_file}" ] && [ -f "${progress_file}" ]; then
        rm -f "${progress_file}"
    fi
}
trap cleanup EXIT

print_error() {
    printf '%s\n' "${SCRIPT_PREFIX} Error: $1" >&2
}

print_status() {
    printf '%s\n' "${SCRIPT_PREFIX} $1"
}

usage() {
    print_error "usage: $0 --profile <profile-name> [--repository-root <path>]"
    print_error "supported profiles: core, core+memory-contract, core+experimental-aligned-expert-packs, core+memory-contract+experimental-aligned-expert-packs"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            if [ "$#" -lt 2 ]; then
                print_error "--profile requires a native build profile name"
                usage
                exit 2
            fi
            profile_name="$2"
            shift 2
            ;;
        --repository-root)
            if [ "$#" -lt 2 ]; then
                print_error "--repository-root requires a directory path"
                usage
                exit 2
            fi
            repository_root="$2"
            shift 2
            ;;
        *)
            print_error "unsupported argument: $1"
            usage
            exit 2
            ;;
    esac
done

if [ -z "${profile_name}" ]; then
    print_error "missing required argument --profile <profile-name>"
    usage
    exit 2
fi

if [ -z "${repository_root}" ]; then
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
fi

if [ ! -d "${repository_root}/crates/runtime-integration/native" ]; then
    print_error "repository root does not contain the native runtime sources: ${repository_root}"
    exit 2
fi

progress_file="$(mktemp "${TMPDIR:-/tmp}/astronomical-native-build-prewarm.XXXXXX")"
: > "${progress_file}"
export ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE="${progress_file}"

( exec tail -f "${progress_file}" ) &
progress_tail_pid=$!
prewarm_started_at="$(date +%s)"
print_status "status=start profile=${profile_name} started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

native_build_exit_code=0
cargo run -p astronomical-native-build-tool -- \
    --profile "${profile_name}" \
    --repository-root "${repository_root}" || native_build_exit_code=$?

prewarm_elapsed_seconds=$(( $(date +%s) - prewarm_started_at ))
if [ "${native_build_exit_code}" -eq 0 ]; then
    print_status "status=success elapsed_seconds=${prewarm_elapsed_seconds}"
else
    print_status "status=failed exit_code=${native_build_exit_code} elapsed_seconds=${prewarm_elapsed_seconds}"
fi

exit "${native_build_exit_code}"
