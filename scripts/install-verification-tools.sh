#!/usr/bin/env sh

# Installs the pinned cargo-about and sccache prebuilt archives from GitHub
# Releases. Hosted CI was spending minutes compiling cargo-about from crates.io
# and brewing sccache; both publishers already ship SHA-256-verified binaries.

set -eu
# pipefail is Bash/Zsh; the subshell probe keeps this script POSIX-runnable.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    set -o pipefail
fi

STARTED_AT_SECONDS=""
TEMPORARY_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -z "${TEMPORARY_DIRECTORY:-}" ]; then
        return
    fi
    case "$TEMPORARY_DIRECTORY" in
        /|.|..)
            print_error "refusing to remove unsafe verification-tools directory"
            return
            ;;
    esac
    if [ -d "$TEMPORARY_DIRECTORY" ]; then
        rm -rf "$TEMPORARY_DIRECTORY"
    fi
}
trap cleanup 0

read_dotted_version_pin() {
    pin_file_path="$1"
    pin_label="$2"
    [ -f "$pin_file_path" ] || {
        print_error "missing ${pin_label} pin file: ${pin_file_path}"
        exit 1
    }
    pinned_version="$(tr -d '[:space:]' < "$pin_file_path")"
    case "$pinned_version" in
        ''|*[!0-9.]*)
            print_error "${pin_label} pin must be a dotted version: ${pin_file_path}"
            exit 1
            ;;
    esac
    printf '%s\n' "$pinned_version"
}

host_rust_target() {
    host_kernel="$(uname -s)"
    host_machine="$(uname -m)"
    case "${host_kernel}:${host_machine}" in
        Darwin:arm64) printf '%s\n' 'aarch64-apple-darwin' ;;
        Darwin:x86_64) printf '%s\n' 'x86_64-apple-darwin' ;;
        Linux:aarch64 | Linux:arm64) printf '%s\n' 'aarch64-unknown-linux-musl' ;;
        Linux:x86_64) printf '%s\n' 'x86_64-unknown-linux-musl' ;;
        *)
            print_error "no prebuilt verification-tool archive mapping for ${host_kernel} ${host_machine}"
            exit 1
            ;;
    esac
}

tool_reports_expected_version() {
    executable_path="$1"
    expected_version_line="$2"
    [ -x "$executable_path" ] || return 1
    installed_version_line="$("$executable_path" --version)"
    [ "$installed_version_line" = "$expected_version_line" ]
}

download_release_file() {
    download_destination_path="$1"
    download_url="$2"
    command -v curl >/dev/null 2>&1 || {
        print_error "curl is required to download pinned verification tools"
        exit 1
    }
    curl -fsSL --retry 3 --retry-delay 1 -o "$download_destination_path" "$download_url" || {
        print_error "failed to download ${download_url}"
        exit 1
    }
}

published_digest_from_sidecar() {
    sidecar_path="$1"
    awk '
        {
            gsub(/\r/, "")
            if ($1 ~ /^[0-9a-fA-F]{64}$/) {
                print $1
                exit
            }
        }
        END {
            if (NR == 0) {
                exit 1
            }
        }
    ' "$sidecar_path"
}

verify_archive_digest() {
    digest_archive_path="$1"
    digest_sidecar_path="$2"
    digest_archive_label="$3"
    command -v shasum >/dev/null 2>&1 || {
        print_error "shasum is required to verify verification-tool archives"
        exit 1
    }
    published_digest="$(published_digest_from_sidecar "$digest_sidecar_path")" || {
        print_error "published SHA-256 sidecar for ${digest_archive_label} is missing a digest"
        exit 1
    }
    actual_digest_line="$(shasum -a 256 "$digest_archive_path")"
    actual_digest="${actual_digest_line%% *}"
    [ "$actual_digest" = "$published_digest" ] || {
        print_error "SHA-256 mismatch for ${digest_archive_label}"
        exit 1
    }
}

extract_named_binary() {
    extract_archive_path="$1"
    extract_binary_name="$2"
    extract_destination_path="$3"
    extraction_directory="${TEMPORARY_DIRECTORY}/extract-${extract_binary_name}"
    rm -rf "$extraction_directory"
    mkdir -p "$extraction_directory"
    tar -xzf "$extract_archive_path" -C "$extraction_directory"
    extracted_binary_path=""
    extracted_binary_count=0
    for candidate_path in "$extraction_directory"/*/"$extract_binary_name" "$extraction_directory/$extract_binary_name"; do
        if [ -f "$candidate_path" ]; then
            extracted_binary_path="$candidate_path"
            extracted_binary_count=$((extracted_binary_count + 1))
        fi
    done
    [ "$extracted_binary_count" -eq 1 ] || {
        print_error "archive for ${extract_binary_name} did not contain exactly one ${extract_binary_name} binary"
        exit 1
    }
    chmod +x "$extracted_binary_path"
    mv "$extracted_binary_path" "$extract_destination_path"
}

install_pinned_binary() {
    binary_name="$1"
    expected_version_line="$2"
    archive_url="$3"
    digest_url="$4"
    install_prefix="$5"
    installed_binary_path="${install_prefix}/${binary_name}"
    if tool_reports_expected_version "$installed_binary_path" "$expected_version_line"; then
        printf '[verification-tools] status=already-installed tool=%s\n' "$binary_name"
        return 0
    fi

    printf '[verification-tools] status=start tool=%s phase=download\n' "$binary_name"
    download_started_at_seconds="$(date +%s)"
    archive_path="${TEMPORARY_DIRECTORY}/${binary_name}.tar.gz"
    sidecar_path="${TEMPORARY_DIRECTORY}/${binary_name}.tar.gz.sha256"
    download_release_file "$archive_path" "$archive_url"
    download_release_file "$sidecar_path" "$digest_url"
    printf '[verification-tools] status=success tool=%s phase=download elapsed_seconds=%s\n' \
        "$binary_name" "$(( $(date +%s) - download_started_at_seconds ))"

    printf '[verification-tools] status=start tool=%s phase=verify-digest\n' "$binary_name"
    verify_started_at_seconds="$(date +%s)"
    verify_archive_digest "$archive_path" "$sidecar_path" "$binary_name"
    printf '[verification-tools] status=success tool=%s phase=verify-digest elapsed_seconds=%s\n' \
        "$binary_name" "$(( $(date +%s) - verify_started_at_seconds ))"

    printf '[verification-tools] status=start tool=%s phase=install\n' "$binary_name"
    install_started_at_seconds="$(date +%s)"
    extract_named_binary "$archive_path" "$binary_name" "$installed_binary_path"
    if ! tool_reports_expected_version "$installed_binary_path" "$expected_version_line"; then
        installed_version_line="$("$installed_binary_path" --version 2>/dev/null || printf '%s\n' 'unreadable')"
        print_error "installed ${binary_name} is ${installed_version_line}; Astronomical requires cargo-about and sccache at the pinned versions (${expected_version_line})"
        exit 1
    fi
    printf '[verification-tools] status=success tool=%s phase=install elapsed_seconds=%s\n' \
        "$binary_name" "$(( $(date +%s) - install_started_at_seconds ))"
}

usage() {
    printf '%s\n' "Usage: scripts/install-verification-tools.sh [--prefix DIR]"
}

main() {
    install_prefix=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --prefix)
                [ "$#" -ge 2 ] || {
                    usage >&2
                    exit 2
                }
                install_prefix="$2"
                shift 2
                ;;
            --help | -h)
                usage
                return 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                usage >&2
                exit 2
                ;;
        esac
    done

    STARTED_AT_SECONDS="$(date +%s)"
    printf '[verification-tools] status=start\n'
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    required_cargo_about_version="$(read_dotted_version_pin \
        "${repository_root}/third-party/cargo-about-version" "cargo-about")"
    required_sccache_version="$(read_dotted_version_pin \
        "${repository_root}/third-party/sccache-version" "sccache")"
    rust_target="$(host_rust_target)"
    if [ -z "$install_prefix" ]; then
        install_prefix="${CARGO_HOME:-${HOME}/.cargo}/bin"
    fi
    mkdir -p "$install_prefix"
    TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-verification-tools.XXXXXX")"

    cargo_about_archive="cargo-about-${required_cargo_about_version}-${rust_target}.tar.gz"
    sccache_archive="sccache-v${required_sccache_version}-${rust_target}.tar.gz"
    install_pinned_binary \
        cargo-about \
        "cargo-about ${required_cargo_about_version}" \
        "https://github.com/EmbarkStudios/cargo-about/releases/download/${required_cargo_about_version}/${cargo_about_archive}" \
        "https://github.com/EmbarkStudios/cargo-about/releases/download/${required_cargo_about_version}/${cargo_about_archive}.sha256" \
        "$install_prefix"
    install_pinned_binary \
        sccache \
        "sccache ${required_sccache_version}" \
        "https://github.com/mozilla/sccache/releases/download/v${required_sccache_version}/${sccache_archive}" \
        "https://github.com/mozilla/sccache/releases/download/v${required_sccache_version}/${sccache_archive}.sha256" \
        "$install_prefix"

    if [ -n "${GITHUB_PATH:-}" ]; then
        printf '%s\n' "$install_prefix" >> "$GITHUB_PATH"
    fi
    printf '[verification-tools] status=success elapsed_seconds=%s cargo_about=%s sccache=%s prefix=%s\n' \
        "$(( $(date +%s) - STARTED_AT_SECONDS ))" \
        "$required_cargo_about_version" \
        "$required_sccache_version" \
        "$install_prefix"
}

main "$@"
