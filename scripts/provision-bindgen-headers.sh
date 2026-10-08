#!/usr/bin/env sh

# Provisions bindgen-ready MLX and MLX-C headers from the verified native
# dependency archive cache, without invoking any CMake build step.
#
# The extraction directory is keyed by the source-only native build identity
# (pins, patches, patch pipeline, store schema), so a pin bump or a patch edit
# invalidates the previous extraction automatically. Extracted MLX-C headers
# can be proven byte-identical to the staged headers of a completed native
# build with --verify-headers, which is the fast surface-diff step of the
# dependency bump process documented in third-party/README.md.
#
# This script never downloads anything. Run scripts/bootstrap-native-dependencies.sh
# first so the verified archives exist in the native dependency cache.
#
# The pipeline itself lives in scripts/internal/bindgen-headers-pipeline.sh;
# this entry owns only the CLI surface.

set -eu

PIPELINE_LIBRARY_PATH="$(dirname -- "$0")/internal/bindgen-headers-pipeline.sh"
# shellcheck source=scripts/internal/bindgen-headers-pipeline.sh
. "$PIPELINE_LIBRARY_PATH"

# CLI-owned state: the pipeline library never reads these two mode flags.
VERIFY_HEADERS_ONLY="false"
PRINT_PATCH_LIST_ONLY="false"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

log_line() {
    printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

require_command() {
    required_command="$1"
    if ! command -v "$required_command" >/dev/null 2>&1; then
        print_error "required command is unavailable: $required_command"
        exit 1
    fi
}

require_absolute_directory_path() {
    directory_path="$1"
    directory_description="$2"
    case "$directory_path" in
        /*) ;;
        *)
            print_error "$directory_description must be an absolute path: $directory_path"
            exit 1
            ;;
    esac
}

print_usage() {
    printf '%s\n' "Usage: scripts/provision-bindgen-headers.sh [options]"
    printf '%s\n' ""
    printf '%s\n' "Extracts the pinned MLX and MLX-C archives from the verified native"
    printf '%s\n' "dependency cache, applies the same patch pipeline as the native build,"
    printf '%s\n' "and publishes them under a pin-keyed directory for bindgen and"
    printf '%s\n' "header-level tooling. Default cache: \$HOME/Library/Caches/Astronomical/native-dependencies"
    printf '%s\n' ""
    printf '%s\n' "Options:"
    printf '%s\n' "  --cache-dir ABSOLUTE_PATH        Read verified archives from this native dependency cache directory."
    printf '%s\n' "  --native-build-store-dir PATH    Native build store for --verify-headers"
    printf '%s\n' "                                   (default: \$ASTRONOMICAL_NATIVE_BUILD_STORE_DIR or \$HOME/Library/Caches/Astronomical/native-builds)."
    printf '%s\n' "  --manifest FILE                  Read the pinned archive manifest from FILE instead of"
    printf '%s\n' "                                   generating it from third-party pins. Used by tooling and contract tests."
    printf '%s\n' "  --patch-list FILE                Read the patch pipeline from FILE (lines: dependency<TAB>patch-path)"
    printf '%s\n' "                                   instead of parsing the native CMakeLists. Used by tooling and contract tests."
    printf '%s\n' "  --profile NAME                   Native build profile for --verify-headers (default: core)."
    printf '%s\n' "  --verify-headers                 Compare extracted MLX-C headers against the staged headers of"
    printf '%s\n' "                                   the completed native build for the current identity and exit."
    printf '%s\n' "  --print-patch-list               Print the parsed patch pipeline (dependency<TAB>patch-path) and exit."
    printf '%s\n' "  --help                           Show this help."
}

parse_arguments() {
    NATIVE_DEPENDENCY_CACHE_DIRECTORY="${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR:-}"
    NATIVE_BUILD_STORE_DIRECTORY="${ASTRONOMICAL_NATIVE_BUILD_STORE_DIR:-}"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --cache-dir)
                [ "$#" -ge 2 ] || { print_error "--cache-dir requires an absolute path"; exit 2; }
                NATIVE_DEPENDENCY_CACHE_DIRECTORY="$2"
                shift 2
                ;;
            --native-build-store-dir)
                [ "$#" -ge 2 ] || { print_error "--native-build-store-dir requires a path"; exit 2; }
                NATIVE_BUILD_STORE_DIRECTORY="$2"
                shift 2
                ;;
            --manifest)
                [ "$#" -ge 2 ] || { print_error "--manifest requires a file path"; exit 2; }
                EXPLICIT_MANIFEST_PATH="$2"
                shift 2
                ;;
            --patch-list)
                [ "$#" -ge 2 ] || { print_error "--patch-list requires a file path"; exit 2; }
                EXPLICIT_PATCH_LIST_PATH="$2"
                shift 2
                ;;
            --profile)
                [ "$#" -ge 2 ] || { print_error "--profile requires a native build profile name"; exit 2; }
                NATIVE_BUILD_PROFILE="$2"
                shift 2
                ;;
            --verify-headers)
                VERIFY_HEADERS_ONLY="true"
                shift
                ;;
            --print-patch-list)
                PRINT_PATCH_LIST_ONLY="true"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                print_usage >&2
                exit 2
                ;;
        esac
    done
}

validate_profile() {
    case "$NATIVE_BUILD_PROFILE" in
        core|core+memory-contract) ;;
        *)
            print_error "unsupported native build profile: $NATIVE_BUILD_PROFILE"
            exit 2
            ;;
    esac
}

configure_directories() {
    if [ -z "$NATIVE_DEPENDENCY_CACHE_DIRECTORY" ]; then
        [ -n "${HOME:-}" ] || {
            print_error "HOME is required when --cache-dir and \$ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR are unset"
            exit 1
        }
        NATIVE_DEPENDENCY_CACHE_DIRECTORY="${HOME}/Library/Caches/Astronomical/native-dependencies"
    fi
    require_absolute_directory_path "$NATIVE_DEPENDENCY_CACHE_DIRECTORY" "native dependency cache directory"
    if [ -z "$NATIVE_BUILD_STORE_DIRECTORY" ]; then
        [ -n "${HOME:-}" ] || {
            print_error "HOME is required when --native-build-store-dir and \$ASTRONOMICAL_NATIVE_BUILD_STORE_DIR are unset"
            exit 1
        }
        NATIVE_BUILD_STORE_DIRECTORY="${HOME}/Library/Caches/Astronomical/native-builds"
    fi
    require_absolute_directory_path "$NATIVE_BUILD_STORE_DIRECTORY" "native build store directory"
}

main() {
    parse_arguments "$@"
    validate_profile

    start_step "validate-inputs"
    require_command cmake
    require_command shasum
    require_command tar
    require_command awk
    resolve_repository_root
    TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-bindgen-headers.XXXXXX")"
    configure_directories
    finish_step "validate-inputs" "success"

    if [ "$PRINT_PATCH_LIST_ONLY" = "true" ]; then
        parse_patch_pipeline
        cat "$EFFECTIVE_PATCH_LIST_PATH"
        return 0
    fi

    if [ "$VERIFY_HEADERS_ONLY" = "true" ]; then
        verify_headers_against_native_build
        return 0
    fi

    provision_headers
}

trap cleanup 0
main "$@"
