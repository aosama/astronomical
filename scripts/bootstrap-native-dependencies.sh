#!/usr/bin/env sh

set -eu

readonly NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE="ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR"
readonly DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX="Library/Caches/Astronomical/native-dependencies"
readonly MAX_ARCHIVE_SIZE_BYTES=268435456
readonly MINIMUM_FREE_SPACE_BYTES=2147483648

NATIVE_DEPENDENCY_CACHE_DIRECTORY=""
IS_VERIFY_ONLY="false"
TEMPORARY_MANIFEST_PATH=""
TEMPORARY_ARCHIVE_PATH=""

print_usage() {
    printf '%s\n' "Usage: scripts/bootstrap-native-dependencies.sh [--cache-dir ABSOLUTE_PATH] [--verify]"
    printf '%s\n' ""
    printf '%s\n' "Downloads and SHA-256-verifies Astronomical's pinned MLX source archives."
    printf '%s\n' "Default native dependency cache: \$HOME/${DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX}"
    printf '%s\n' ""
    printf '%s\n' "--cache-dir ABSOLUTE_PATH  Store or verify archives in this native dependency cache directory."
    printf '%s\n' "--verify                   Verify cached archives without downloading."
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

step_started_at_seconds=0

start_step() {
    step_name="$1"
    step_started_at_seconds="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name"
}

finish_step() {
    step_name="$1"
    step_status="$2"
    step_finished_at_seconds="$(date +%s)"
    step_elapsed_seconds=$((step_finished_at_seconds - step_started_at_seconds))
    printf '%s step=%s status=%s elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" "$step_status" "$step_elapsed_seconds"
}

cleanup() {
    if [ -n "${TEMPORARY_MANIFEST_PATH:-}" ] && [ -f "$TEMPORARY_MANIFEST_PATH" ]; then
        rm -f -- "$TEMPORARY_MANIFEST_PATH"
    fi

    if [ -n "${TEMPORARY_ARCHIVE_PATH:-}" ] && [ -f "$TEMPORARY_ARCHIVE_PATH" ]; then
        rm -f -- "$TEMPORARY_ARCHIVE_PATH"
    fi
}
trap cleanup 0

require_command() {
    required_command="$1"
    if ! command -v "$required_command" >/dev/null 2>&1; then
        print_error "required command is unavailable: $required_command"
        exit 1
    fi
}

require_absolute_directory_path() {
    directory_path="$1"
    case "$directory_path" in
        /*) ;;
        *)
            print_error "native dependency cache directory must be an absolute path: $directory_path"
            exit 1
            ;;
    esac
}

configure_native_dependency_cache_directory() {
    if [ -n "$NATIVE_DEPENDENCY_CACHE_DIRECTORY" ]; then
        require_absolute_directory_path "$NATIVE_DEPENDENCY_CACHE_DIRECTORY"
        return
    fi

    if [ -z "${HOME:-}" ]; then
        print_error "HOME is required when --cache-dir and $NATIVE_DEPENDENCY_CACHE_DIRECTORY_VARIABLE are unset"
        exit 1
    fi

    NATIVE_DEPENDENCY_CACHE_DIRECTORY="${HOME}/${DEFAULT_NATIVE_DEPENDENCY_CACHE_DIRECTORY_SUFFIX}"
    require_absolute_directory_path "$NATIVE_DEPENDENCY_CACHE_DIRECTORY"
}

ensure_free_space() {
    native_dependency_cache_file_system_statistics="$(stat -f '%a %k' "$NATIVE_DEPENDENCY_CACHE_DIRECTORY")"
    IFS=' ' read -r available_file_system_blocks file_system_block_size_bytes extra_file_system_statistics <<EOF
$native_dependency_cache_file_system_statistics
EOF
    if [ -n "${extra_file_system_statistics:-}" ]; then
        print_error "could not parse native dependency cache filesystem statistics"
        exit 1
    fi
    case "$available_file_system_blocks" in
        ''|*[!0-9]*)
            print_error "could not determine available native dependency cache filesystem blocks"
            exit 1
            ;;
    esac
    case "$file_system_block_size_bytes" in
        ''|*[!0-9]*|0)
            print_error "could not determine native dependency cache filesystem block size"
            exit 1
            ;;
    esac

    required_file_system_blocks=$((MINIMUM_FREE_SPACE_BYTES / file_system_block_size_bytes + 1))
    if [ "$available_file_system_blocks" -lt "$required_file_system_blocks" ]; then
        print_error "native dependency cache filesystem needs at least ${MINIMUM_FREE_SPACE_BYTES} free bytes before downloading"
        exit 1
    fi
}

sha256_matches() (
    candidate_archive_path="$1"
    expected_sha256="$2"
    if [ ! -f "$candidate_archive_path" ] || [ -L "$candidate_archive_path" ]; then
        return 1
    fi

    actual_sha256_line="$(shasum -a 256 "$candidate_archive_path")"
    actual_sha256="${actual_sha256_line%% *}"
    [ "$actual_sha256" = "$expected_sha256" ]
)

validate_manifest_field() {
    manifest_field_name="$1"
    manifest_field_text="$2"
    case "$manifest_field_text" in
        ''|*'|'*|*'\n'*|*'\r'*)
            print_error "native dependency manifest has an invalid $manifest_field_name"
            exit 1
            ;;
    esac
}

provision_archive() {
    archive_file_name="$1"
    archive_url="$2"
    archive_sha256="$3"
    dependency_description="$4"
    archive_path="${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/${archive_file_name}"

    if [ -e "$archive_path" ] && [ ! -f "$archive_path" ]; then
        print_error "native dependency cache path is not a regular file: $archive_path"
        exit 1
    fi

    if sha256_matches "$archive_path" "$archive_sha256"; then
        printf '%s archive=%s status=verified-cached\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$dependency_description"
        return
    fi

    if [ "$IS_VERIFY_ONLY" = "true" ]; then
        print_error "missing or invalid $dependency_description archive: $archive_path"
        exit 1
    fi

    TEMPORARY_ARCHIVE_PATH="$(mktemp "${NATIVE_DEPENDENCY_CACHE_DIRECTORY}/.${archive_file_name}.download.XXXXXX")"
    if ! curl --fail --location --proto '=https' --tlsv1.2 --retry 3 --retry-all-errors \
        --connect-timeout 30 --max-time 900 --max-filesize "$MAX_ARCHIVE_SIZE_BYTES" \
        --output "$TEMPORARY_ARCHIVE_PATH" "$archive_url"; then
        print_error "failed to download $dependency_description from $archive_url"
        exit 1
    fi

    downloaded_size_report="$(wc -c < "$TEMPORARY_ARCHIVE_PATH")"
    IFS=' ' read -r downloaded_size_bytes extra_download_size_report <<EOF
$downloaded_size_report
EOF
    if [ -n "${extra_download_size_report:-}" ]; then
        print_error "could not parse downloaded size for $dependency_description"
        exit 1
    fi
    case "$downloaded_size_bytes" in
        ''|*[!0-9]*)
            print_error "could not determine downloaded size for $dependency_description"
            exit 1
            ;;
    esac
    if [ "$downloaded_size_bytes" -gt "$MAX_ARCHIVE_SIZE_BYTES" ]; then
        print_error "$dependency_description exceeds the ${MAX_ARCHIVE_SIZE_BYTES}-byte archive limit"
        exit 1
    fi

    if ! sha256_matches "$TEMPORARY_ARCHIVE_PATH" "$archive_sha256"; then
        print_error "$dependency_description SHA-256 verification failed"
        exit 1
    fi

    chmod 600 "$TEMPORARY_ARCHIVE_PATH"
    mv -f "$TEMPORARY_ARCHIVE_PATH" "$archive_path"
    TEMPORARY_ARCHIVE_PATH=""
    printf '%s archive=%s status=downloaded-and-verified bytes=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$dependency_description" "$downloaded_size_bytes"
}

parse_arguments() {
    NATIVE_DEPENDENCY_CACHE_DIRECTORY="${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR:-}"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --cache-dir)
                if [ "$#" -lt 2 ]; then
                    print_error "--cache-dir requires an absolute path"
                    exit 1
                fi
                NATIVE_DEPENDENCY_CACHE_DIRECTORY="$2"
                shift 2
                ;;
            --verify)
                IS_VERIFY_ONLY="true"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                print_usage >&2
                exit 1
                ;;
        esac
    done
}

main() {
    parse_arguments "$@"

    start_step "validate-inputs"
    require_command cmake
    require_command shasum
    if [ "$IS_VERIFY_ONLY" != "true" ]; then
        require_command curl
    fi
    configure_native_dependency_cache_directory
    mkdir -p "$NATIVE_DEPENDENCY_CACHE_DIRECTORY"
    chmod 700 "$NATIVE_DEPENDENCY_CACHE_DIRECTORY"
    if [ "$IS_VERIFY_ONLY" != "true" ]; then
        ensure_free_space
    fi
    finish_step "validate-inputs" "success"

    start_step "read-pinned-manifest"
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    TEMPORARY_MANIFEST_PATH="$(mktemp "${TMPDIR:-/tmp}/astronomical-native-dependencies.XXXXXX")"
    cmake \
        "-DASTRONOMICAL_NATIVE_DEPENDENCY_MANIFEST_PATH=${TEMPORARY_MANIFEST_PATH}" \
        -P "${repository_root}/third-party/native-dependency-manifest.cmake"
    finish_step "read-pinned-manifest" "success"

    start_step "verify-or-download-archives"
    ulimit -f $((MAX_ARCHIVE_SIZE_BYTES / 512))
    dependency_count=0
    while IFS='|' read -r archive_file_name archive_url archive_sha256 dependency_description; do
        validate_manifest_field "archive file name" "$archive_file_name"
        validate_manifest_field "archive URL" "$archive_url"
        validate_manifest_field "archive SHA-256" "$archive_sha256"
        validate_manifest_field "dependency description" "$dependency_description"
        case "$archive_file_name" in
            *[!A-Za-z0-9._-]*|'' )
                print_error "native dependency manifest has an unsafe archive file name"
                exit 1
                ;;
        esac
        case "$archive_url" in
            https://*) ;;
            *)
                print_error "native dependency manifest requires an HTTPS archive URL"
                exit 1
                ;;
        esac
        case "$archive_sha256" in
            *[!0123456789abcdef]*|'' )
                print_error "native dependency manifest has a non-lowercase SHA-256"
                exit 1
                ;;
        esac
        if [ "${#archive_sha256}" -ne 64 ]; then
            print_error "native dependency manifest has an invalid SHA-256 length"
            exit 1
        fi
        provision_archive "$archive_file_name" "$archive_url" "$archive_sha256" "$dependency_description"
        dependency_count=$((dependency_count + 1))
    done < "$TEMPORARY_MANIFEST_PATH"
    if [ "$dependency_count" -ne 5 ]; then
        print_error "native dependency manifest must contain exactly five archives"
        exit 1
    fi
    finish_step "verify-or-download-archives" "success"

    printf '%s native_dependency_cache_directory=%s status=ready\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$NATIVE_DEPENDENCY_CACHE_DIRECTORY"
}

main "$@"
