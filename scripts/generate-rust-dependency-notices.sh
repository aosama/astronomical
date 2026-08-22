#!/usr/bin/env sh

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
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

    if ! command -v cargo-about >/dev/null 2>&1; then
        print_error "cargo-about is required; install Homebrew cargo-about"
        exit 1
    fi
    if ! command -v perl >/dev/null 2>&1; then
        print_error "perl is required to normalize generated license text"
        exit 1
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    generated_notices_path="${repository_root}/third-party/RUST_DEPENDENCY_NOTICES"
    generation_destination_path="$generated_notices_path"
    temporary_notices_path=""

    if [ "$check_only" = true ]; then
        temporary_notices_path="$(mktemp "${TMPDIR:-/tmp}/astronomical-rust-notices.XXXXXX")"
        trap 'rm -f "${temporary_notices_path:-}"' 0
        generation_destination_path="$temporary_notices_path"
    fi

    printf '[rust-dependency-notices] status=generating destination=%s\n' "$generation_destination_path"
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
