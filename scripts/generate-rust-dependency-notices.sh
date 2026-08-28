#!/usr/bin/env sh

# Regenerates or checks third-party/RUST_DEPENDENCY_NOTICES with the pinned
# cargo-about. Version drift here is a tool change, not an Astronomical crate change.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

read_pinned_cargo_about_version() {
    pin_file_path="$1"
    [ -f "$pin_file_path" ] || {
        print_error "missing cargo-about pin file: ${pin_file_path}"
        exit 1
    }
    pinned_cargo_about_version="$(tr -d '[:space:]' < "$pin_file_path")"
    case "$pinned_cargo_about_version" in
        ''|*[!0-9.]*)
            print_error "cargo-about pin must be a dotted version: ${pin_file_path}"
            exit 1
            ;;
    esac
    printf '%s\n' "$pinned_cargo_about_version"
}

require_pinned_cargo_about() {
    required_cargo_about_version="$1"
    if ! command -v cargo-about >/dev/null 2>&1; then
        print_error "cargo-about ${required_cargo_about_version} is required; install that exact version"
        exit 1
    fi
    installed_cargo_about_version="$(cargo-about --version)"
    expected_cargo_about_version="cargo-about ${required_cargo_about_version}"
    if [ "$installed_cargo_about_version" != "$expected_cargo_about_version" ]; then
        print_error "notices generation requires cargo-about ${required_cargo_about_version}; found ${installed_cargo_about_version}"
        exit 1
    fi
}

main() {
    check_only=false
    if [ "$#" -gt 1 ]; then
        print_error "usage: scripts/generate-rust-dependency-notices.sh [--check]"
        exit 2
    fi
    if [ "$#" -eq 1 ]; then
        if [ "$1" != "--check" ]; then
            print_error "unrecognized argument: $1"
            exit 2
        fi
        check_only=true
    fi

    if ! command -v perl >/dev/null 2>&1; then
        print_error "perl is required to normalize generated license text"
        exit 1
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    required_cargo_about_version="$(read_pinned_cargo_about_version "${repository_root}/third-party/cargo-about-version")"
    require_pinned_cargo_about "$required_cargo_about_version"
    generated_notices_path="${repository_root}/third-party/RUST_DEPENDENCY_NOTICES"
    generation_destination_path="$generated_notices_path"
    temporary_notices_path=""

    if [ "$check_only" = true ]; then
        temporary_notices_path="$(mktemp "${TMPDIR:-/tmp}/astronomical-rust-notices.XXXXXX")"
        trap 'rm -f "${temporary_notices_path:-}"' 0
        generation_destination_path="$temporary_notices_path"
    fi

    printf '[rust-dependency-notices] status=generating destination=%s cargo_about=%s\n' \
        "$generation_destination_path" "$required_cargo_about_version"
    cargo about generate \
        --workspace \
        --all-features \
        --locked \
        --fail \
        --config "${repository_root}/third-party/about.toml" \
        --output-file "$generation_destination_path" \
        "${repository_root}/third-party/rust-dependency-notices.hbs"
    # Upstream license files can contain invisible line-end padding; stripping it keeps the generated artifact reviewable and diff-check clean without changing license wording.
    perl -0pi -e 's/\r\n/\n/g; s/[ \t]+(?=\n)//g; s/\n+\z/\n/' "$generation_destination_path"

    if [ "$check_only" = true ]; then
        if ! cmp -s "$temporary_notices_path" "$generated_notices_path"; then
            print_error "third-party/RUST_DEPENDENCY_NOTICES is stale; regenerate it"
            exit 1
        fi
        printf '%s\n' '[rust-dependency-notices] status=current'
    else
        printf '%s\n' '[rust-dependency-notices] status=updated'
    fi
}

main "$@"
