#!/usr/bin/env sh

# Produces the compatibility identity shared by the native build store and CI.
# Source-only mode remains portable so the lightweight classification job can
# coordinate unchanged event topology before a macOS runner is allocated.

set -eu

TEMPORARY_DIRECTORY=""
SOURCE_ONLY="false"
NATIVE_BUILD_PROFILE=""
REPOSITORY_ROOT_ARGUMENT=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${TEMPORARY_DIRECTORY:-}" ] && [ -d "$TEMPORARY_DIRECTORY" ]; then
        case "$TEMPORARY_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe native identity directory" ;;
            *) rm -rf "$TEMPORARY_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

print_usage() {
    print_error "Usage: scripts/native-build-cache-fingerprint.sh [--source-only] --profile PROFILE [repository-root]"
}

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --source-only)
                SOURCE_ONLY="true"
                shift
                ;;
            --profile)
                [ "$#" -ge 2 ] || {
                    print_usage
                    exit 2
                }
                NATIVE_BUILD_PROFILE="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            --*)
                print_error "unsupported native identity option: $1"
                exit 2
                ;;
            *)
                [ -z "$REPOSITORY_ROOT_ARGUMENT" ] || {
                    print_usage
                    exit 2
                }
                REPOSITORY_ROOT_ARGUMENT="$1"
                shift
                ;;
        esac
    done
    case "$NATIVE_BUILD_PROFILE" in
        core|core+memory-contract|core+experimental-aligned-expert-packs|core+memory-contract+experimental-aligned-expert-packs) ;;
        '')
            print_error "--profile is required"
            exit 2
            ;;
        *)
            print_error "unsupported native build profile: ${NATIVE_BUILD_PROFILE}"
            exit 2
            ;;
    esac
}

require_command() {
    command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || {
        print_error "${command_name} is required to fingerprint native build inputs"
        exit 2
    }
}

resolve_repository_root() {
    if [ -n "$REPOSITORY_ROOT_ARGUMENT" ]; then
        repository_root_candidate="$REPOSITORY_ROOT_ARGUMENT"
    else
        repository_root_candidate="$(dirname -- "$0")/.."
    fi
    repository_root="$(CDPATH='' cd -- "$repository_root_candidate" && pwd -P)" || {
        print_error "repository root is unavailable: ${repository_root_candidate}"
        exit 2
    }
    git -C "$repository_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        print_error "repository root is not a Git worktree: ${repository_root}"
        exit 2
    }
}

append_source_identity() {
    tracked_paths_file="${TEMPORARY_DIRECTORY}/tracked-paths"
    git -C "$repository_root" ls-files --cached --others --exclude-standard -- \
        crates/runtime-integration/build.rs \
        crates/runtime-integration/build_bindings.rs \
        crates/runtime-integration/build_native_linking.rs \
        crates/runtime-integration/build_native_store.rs \
        crates/runtime-integration/build_native_store_manifest.rs \
        crates/runtime-integration/native-build-store-schema-version \
        crates/runtime-integration/native \
        scripts/native-build-cache-fingerprint.sh \
        third-party/native-dependency-manifest.cmake \
        third-party/pins \
        third-party/patches \
        > "$tracked_paths_file"
    [ -s "$tracked_paths_file" ] || {
        print_error "no tracked native build inputs were found"
        exit 1
    }
    LC_ALL=C sort "$tracked_paths_file" -o "$tracked_paths_file"
    while IFS= read -r tracked_path; do
        source_path="${repository_root}/${tracked_path}"
        [ -f "$source_path" ] || {
            print_error "tracked native build input is unavailable: ${tracked_path}"
            exit 1
        }
        source_digest_output="$(shasum -a 256 "$source_path")"
        source_digest="${source_digest_output%% *}"
        printf 'source\t%s\t%s\n' "$source_digest" "$tracked_path" >> "$identity_manifest"
    done < "$tracked_paths_file"
}

capture_compatibility_input() {
    input_name="$1"
    override_variable_name="$2"
    shift 2
    input_path="${TEMPORARY_DIRECTORY}/${input_name}"
    case "$override_variable_name" in
        ASTRONOMICAL_NATIVE_IDENTITY_XCODE) override_text="${ASTRONOMICAL_NATIVE_IDENTITY_XCODE:-}" ;;
        ASTRONOMICAL_NATIVE_IDENTITY_SDK) override_text="${ASTRONOMICAL_NATIVE_IDENTITY_SDK:-}" ;;
        ASTRONOMICAL_NATIVE_IDENTITY_CLANG) override_text="${ASTRONOMICAL_NATIVE_IDENTITY_CLANG:-}" ;;
        ASTRONOMICAL_NATIVE_IDENTITY_CMAKE) override_text="${ASTRONOMICAL_NATIVE_IDENTITY_CMAKE:-}" ;;
        ASTRONOMICAL_NATIVE_IDENTITY_RUSTC) override_text="${ASTRONOMICAL_NATIVE_IDENTITY_RUSTC:-}" ;;
        *)
            print_error "unsupported compatibility override: ${override_variable_name}"
            exit 2
            ;;
    esac
    if [ -n "$override_text" ]; then
        printf '%s\n' "$override_text" > "$input_path"
    else
        "$@" > "$input_path"
    fi
    [ -s "$input_path" ] || {
        print_error "native compatibility input is empty: ${input_name}"
        exit 1
    }
    input_digest_output="$(shasum -a 256 "$input_path")"
    input_digest="${input_digest_output%% *}"
    printf 'compatibility\t%s\t%s\n' "$input_name" "$input_digest" >> "$identity_manifest"
}

capture_xcode_identity() {
    xcodebuild -version
}

capture_sdk_identity() {
    sdk_version="$(xcrun --sdk macosx --show-sdk-version)"
    sdk_build_version="$(xcrun --sdk macosx --show-sdk-build-version)"
    printf 'version=%s\n' "$sdk_version"
    printf 'build=%s\n' "$sdk_build_version"
}

capture_clang_identity() {
    clang_version="$(xcrun clang -dumpversion)"
    clang_target="$(xcrun clang -dumpmachine)"
    printf 'version=%s\n' "$clang_version"
    printf 'target=%s\n' "$clang_target"
}

capture_cmake_identity() {
    cmake --version
}

capture_rustc_identity() {
    rustc -vV
}

append_compatibility_identity() {
    target_identity="${ASTRONOMICAL_NATIVE_IDENTITY_TARGET:-${TARGET:-}}"
    build_type="${ASTRONOMICAL_NATIVE_BUILD_TYPE:-Release}"
    [ -n "$target_identity" ] || {
        print_error "TARGET or ASTRONOMICAL_NATIVE_IDENTITY_TARGET is required"
        exit 2
    }
    [ -n "$build_type" ] || {
        print_error "ASTRONOMICAL_NATIVE_BUILD_TYPE must not be empty"
        exit 2
    }
    [ "$build_type" = "Release" ] || {
        print_error "only the Release native build type is supported"
        exit 2
    }
    printf 'compatibility\ttarget\t%s\n' "$target_identity" >> "$identity_manifest"
    printf 'compatibility\tbuild-type\t%s\n' "$build_type" >> "$identity_manifest"
    capture_compatibility_input xcode ASTRONOMICAL_NATIVE_IDENTITY_XCODE capture_xcode_identity
    capture_compatibility_input sdk ASTRONOMICAL_NATIVE_IDENTITY_SDK capture_sdk_identity
    capture_compatibility_input clang ASTRONOMICAL_NATIVE_IDENTITY_CLANG capture_clang_identity
    capture_compatibility_input cmake ASTRONOMICAL_NATIVE_IDENTITY_CMAKE capture_cmake_identity
    capture_compatibility_input rustc ASTRONOMICAL_NATIVE_IDENTITY_RUSTC capture_rustc_identity
}

main() {
    parse_arguments "$@"
    for required_command in git mktemp shasum sort; do
        require_command "$required_command"
    done
    resolve_repository_root
    TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-native-identity.XXXXXX")"
    identity_manifest="${TEMPORARY_DIRECTORY}/identity-manifest"
    started_at_seconds="$(date +%s)"
    printf '[native-build-cache-fingerprint] status=start source_only=%s profile=%s\n' \
        "$SOURCE_ONLY" "$NATIVE_BUILD_PROFILE" >&2
    printf 'schema\t2\nprofile\t%s\n' "$NATIVE_BUILD_PROFILE" > "$identity_manifest"
    append_source_identity
    if [ "$SOURCE_ONLY" != "true" ]; then
        append_compatibility_identity
    fi
    identity_digest_output="$(shasum -a 256 "$identity_manifest")"
    native_build_identity="${identity_digest_output%% *}"
    case "$native_build_identity" in
        ????????-*)
            print_error "SHA-256 utility returned an invalid native build identity"
            exit 1
            ;;
        *[!0-9a-f]*|'')
            print_error "SHA-256 utility returned an invalid native build identity"
            exit 1
            ;;
    esac
    [ "${#native_build_identity}" -eq 64 ] || {
        print_error "SHA-256 utility returned an invalid native build identity length"
        exit 1
    }
    printf '%s\n' "$native_build_identity"
    printf '[native-build-cache-fingerprint] status=success elapsed_seconds=%s\n' \
        "$(( $(date +%s) - started_at_seconds ))" >&2
}

main "$@"
