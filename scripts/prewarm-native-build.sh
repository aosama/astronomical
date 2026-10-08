#!/usr/bin/env sh

# Builds Astronomical's pinned native runtime ahead of Cargo so the native
# CMake compile never overlaps Rust compilation. CI and the direct verification
# the macOS app build all call this before their first Cargo step; the later
# build scripts then find the native build store warm and skip CMake entirely.
#
# The native build tool streams progress into a file (Cargo captures build
# script stderr), so this wrapper tails that file to keep live output visible.

set -eu

readonly SCRIPT_PREFIX="[prewarm-native-build]"

profile_names=""
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
    print_error "supported profiles: core, core+memory-contract"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            if [ "$#" -lt 2 ]; then
                print_error "--profile requires a native build profile name"
                usage
                exit 2
            fi
            profile_names="${profile_names:+${profile_names} }$2"
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

if [ -z "${profile_names}" ]; then
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
print_status "status=start profiles=${profile_names} started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Provision the bindgen header extraction before anything builds. The
# MLX-C bindings crate's build script resolves its headers from this
# extraction, and the step is idempotent: a warm cache verifies in about a
# second and never touches the network.
print_status "status=provision-bindgen-headers start"
provision_started_at="$(date +%s)"
if ! sh "${repository_root}/scripts/provision-bindgen-headers.sh"; then
    print_error "bindgen header provisioning failed; run scripts/bootstrap-native-dependencies.sh if the archive cache is empty"
    exit 1
fi
print_status "status=provision-bindgen-headers success elapsed_seconds=$(( $(date +%s) - provision_started_at ))"

native_build_exit_code=0
for profile_name in ${profile_names}; do
    profile_started_at="$(date +%s)"
    print_status "status=native-build start profile=${profile_name}"
    cargo run -p astronomical-native-build-tool -- \
        --profile "${profile_name}" \
        --repository-root "${repository_root}" || {
        native_build_exit_code=$?
        break
    }
    print_status \
        "status=native-build success profile=${profile_name} elapsed_seconds=$(( $(date +%s) - profile_started_at ))"
done

prewarm_elapsed_seconds=$(( $(date +%s) - prewarm_started_at ))
if [ "${native_build_exit_code}" -eq 0 ]; then
    print_status "status=success elapsed_seconds=${prewarm_elapsed_seconds}"
else
    print_status "status=failed exit_code=${native_build_exit_code} elapsed_seconds=${prewarm_elapsed_seconds}"
fi

exit "${native_build_exit_code}"
