#!/usr/bin/env sh

# Regenerates or checks third-party/RUST_DEPENDENCY_NOTICES with the pinned
# cargo-about. Version drift here is a tool change, not an Astronomical crate change.
# --check skips harvest when the committed input/notices digest still matches.

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

list_notices_input_paths() {
    repository_root="$1"
    printf '%s\n' \
        Cargo.lock \
        Cargo.toml \
        third-party/about.toml \
        third-party/rust-dependency-notices.hbs \
        third-party/cargo-about-version
    for source_tree in apps crates experimental; do
        [ -d "${repository_root}/${source_tree}" ] || continue
        (
            CDPATH='' cd -- "${repository_root}/${source_tree}" || exit 1
            find . -name target -prune -o -name Cargo.toml -print
            find . -name target -prune -o \( -name LICENSE -o -name LICENSE.md -o -name COPYING \) -print
        ) | sed "s#^\./#${source_tree}/#"
    done
}

hash_file_sha256() {
    file_path="$1"
    command -v shasum >/dev/null 2>&1 || {
        print_error "shasum is required to digest dependency notices inputs"
        exit 1
    }
    digest_line="$(shasum -a 256 "$file_path")"
    printf '%s\n' "${digest_line%% *}"
}

compute_notices_input_digest() {
    repository_root="$1"
    digest_material="$(
        list_notices_input_paths "$repository_root" | LC_ALL=C sort -u | while IFS= read -r relative_path; do
            [ -n "$relative_path" ] || continue
            input_path="${repository_root}/${relative_path}"
            [ -f "$input_path" ] || continue
            printf '%s  %s\n' "$(hash_file_sha256 "$input_path")" "$relative_path"
        done
    )"
    printf '%s\n' "$digest_material" | shasum -a 256 | awk '{print $1}'
}

write_notices_digest() {
    digest_path="$1"
    input_digest="$2"
    notices_digest="$3"
    printf 'input_sha256=%s\nnotices_sha256=%s\n' "$input_digest" "$notices_digest" > "$digest_path"
}

read_digest_field() {
    digest_path="$1"
    field_name="$2"
    awk -F= -v field="$field_name" '$1 == field { print $2; found = 1 } END { exit found ? 0 : 1 }' "$digest_path"
}

notices_digest_is_current() {
    digest_path="$1"
    notices_path="$2"
    expected_input_digest="$3"
    [ -f "$digest_path" ] || return 1
    [ -f "$notices_path" ] || return 1
    recorded_input_digest="$(read_digest_field "$digest_path" input_sha256)" || return 1
    recorded_notices_digest="$(read_digest_field "$digest_path" notices_sha256)" || return 1
    actual_notices_digest="$(hash_file_sha256 "$notices_path")"
    [ "$recorded_input_digest" = "$expected_input_digest" ] || return 1
    [ "$recorded_notices_digest" = "$actual_notices_digest" ] || return 1
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

    started_at_seconds="$(date +%s)"
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    required_cargo_about_version="$(read_pinned_cargo_about_version "${repository_root}/third-party/cargo-about-version")"
    require_pinned_cargo_about "$required_cargo_about_version"
    generated_notices_path="${repository_root}/third-party/RUST_DEPENDENCY_NOTICES"
    digest_path="${repository_root}/third-party/RUST_DEPENDENCY_NOTICES.digest"
    input_digest="$(compute_notices_input_digest "$repository_root")"

    if [ "$check_only" = true ] && notices_digest_is_current "$digest_path" "$generated_notices_path" "$input_digest"; then
        printf '[rust-dependency-notices] status=current reason=input-digest elapsed_seconds=%s cargo_about=%s\n' \
            "$(( $(date +%s) - started_at_seconds ))" "$required_cargo_about_version"
        return 0
    fi

    if ! command -v perl >/dev/null 2>&1; then
        print_error "perl is required to normalize generated license text"
        exit 1
    fi

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
        if ! notices_digest_is_current "$digest_path" "$generated_notices_path" "$input_digest"; then
            print_error "third-party/RUST_DEPENDENCY_NOTICES.digest is stale; run scripts/generate-rust-dependency-notices.sh"
            exit 1
        fi
        printf '[rust-dependency-notices] status=current reason=harvest elapsed_seconds=%s cargo_about=%s\n' \
            "$(( $(date +%s) - started_at_seconds ))" "$required_cargo_about_version"
        return 0
    fi

    notices_digest="$(hash_file_sha256 "$generated_notices_path")"
    write_notices_digest "$digest_path" "$input_digest" "$notices_digest"
    printf '[rust-dependency-notices] status=updated elapsed_seconds=%s cargo_about=%s\n' \
        "$(( $(date +%s) - started_at_seconds ))" "$required_cargo_about_version"
}

main "$@"
