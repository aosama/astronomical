#!/usr/bin/env sh

# Proves --check skips cargo-about harvest when the committed digest still
# matches, and harvests when that digest is no longer current.

set -eu

SANDBOX_DIRECTORY=""
DIGEST_BACKUP_PATH=""
DIGEST_PATH=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

restore_digest() {
    if [ -n "${DIGEST_BACKUP_PATH:-}" ] && [ -f "$DIGEST_BACKUP_PATH" ] && [ -n "${DIGEST_PATH:-}" ]; then
        mv "$DIGEST_BACKUP_PATH" "$DIGEST_PATH"
    fi
}

cleanup() {
    restore_digest
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe notices-digest sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

write_executable() {
    executable_path="$1"
    cat > "$executable_path"
    chmod +x "$executable_path"
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    notices_script="${repository_root}/scripts/generate-rust-dependency-notices.sh"
    DIGEST_PATH="${repository_root}/third-party/RUST_DEPENDENCY_NOTICES.digest"
    [ -f "$DIGEST_PATH" ] || {
        print_error "missing ${DIGEST_PATH}; run scripts/generate-rust-dependency-notices.sh"
        exit 1
    }
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-notices-digest.XXXXXX")"
    fake_command_directory="${SANDBOX_DIRECTORY}/bin"
    generate_log="${SANDBOX_DIRECTORY}/cargo-about-generate.log"
    mkdir -p "$fake_command_directory"

    printf '%s\n' '[rust-dependency-notices-digest] case=skip-matching-digest status=start'
    write_executable "${fake_command_directory}/cargo-about" <<EOF
#!/usr/bin/env sh
set -eu
if [ "\${1:-}" = "--version" ]; then
    printf '%s\\n' "cargo-about 0.9.2"
    exit 0
fi
printf '%s\\n' "\$*" >> "${generate_log}"
printf '%s\\n' "Error: cargo about generate must not run when the notices digest matches" >&2
exit 1
EOF
    : > "$generate_log"
    skip_output="$(
        PATH="${fake_command_directory}:${PATH}" "$notices_script" --check
    )"
    case "$skip_output" in
        *"reason=input-digest"*) ;;
        *)
            print_error "matching digest still harvested licenses: ${skip_output}"
            exit 1
            ;;
    esac
    [ ! -s "$generate_log" ] || {
        print_error "matching digest invoked cargo about generate"
        exit 1
    }
    printf '%s\n' '[rust-dependency-notices-digest] case=skip-matching-digest status=success'

    printf '%s\n' '[rust-dependency-notices-digest] case=harvest-when-digest-mismatches status=start'
    DIGEST_BACKUP_PATH="${SANDBOX_DIRECTORY}/RUST_DEPENDENCY_NOTICES.digest.backup"
    cp "$DIGEST_PATH" "$DIGEST_BACKUP_PATH"
    printf '%s\n' 'input_sha256=0000000000000000000000000000000000000000000000000000000000000000' \
        'notices_sha256=0000000000000000000000000000000000000000000000000000000000000000' \
        > "$DIGEST_PATH"
    mismatch_output="$(
        PATH="${fake_command_directory}:${PATH}" "$notices_script" --check 2>&1
    )" || mismatch_exit_status=$?
    restore_digest
    DIGEST_BACKUP_PATH=""
    [ "${mismatch_exit_status:-0}" -ne 0 ] || {
        print_error "a broken digest skipped harvest"
        exit 1
    }
    grep -F 'generate' "$generate_log" >/dev/null || {
        print_error "a broken digest did not invoke cargo about generate: ${mismatch_output}"
        exit 1
    }
    printf '%s\n' '[rust-dependency-notices-digest] case=harvest-when-digest-mismatches status=success'
    printf '%s\n' '[rust-dependency-notices-digest] status=success'
}

main "$@"
