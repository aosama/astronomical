#!/usr/bin/env sh

# Proves cargo-about is pinned to the repository version on every notices path.

set -eu

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe notices-contract sandbox" ;;
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
    pinned_cargo_about_version="$(tr -d '[:space:]' < "${repository_root}/third-party/cargo-about-version")"
    [ "$pinned_cargo_about_version" = "0.9.2" ] || {
        print_error "third-party/cargo-about-version must pin 0.9.2"
        exit 1
    }
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-notices-contract.XXXXXX")"
    fake_command_directory="${SANDBOX_DIRECTORY}/bin"
    mkdir -p "$fake_command_directory"

    printf '%s\n' '[rust-dependency-notices-contract] case=reject-missing-cargo-about status=start'
    write_executable "${fake_command_directory}/perl" <<'PERL'
#!/usr/bin/env sh
exit 0
PERL
    missing_output="$(
        PATH="${fake_command_directory}:/usr/bin:/bin" "$notices_script" --check 2>&1
    )" || missing_exit_status=$?
    [ "${missing_exit_status:-0}" -ne 0 ] || {
        print_error "notices generation accepted a PATH without cargo-about"
        exit 1
    }
    case "$missing_output" in
        *"cargo-about ${pinned_cargo_about_version} is required"*) ;;
        *)
            print_error "missing cargo-about did not name the pinned version: ${missing_output}"
            exit 1
            ;;
    esac
    printf '%s\n' '[rust-dependency-notices-contract] case=reject-missing-cargo-about status=success'

    printf '%s\n' '[rust-dependency-notices-contract] case=reject-unpinned-cargo-about status=start'
    write_executable "${fake_command_directory}/cargo-about" <<'CARGO_ABOUT'
#!/usr/bin/env sh
set -eu
if [ "${1:-}" = "--version" ]; then
    printf '%s\n' "cargo-about 0.8.0"
    exit 0
fi
exit 1
CARGO_ABOUT
    unpinned_output="$(
        PATH="${fake_command_directory}:${PATH}" "$notices_script" --check 2>&1
    )" || unpinned_exit_status=$?
    [ "${unpinned_exit_status:-0}" -ne 0 ] || {
        print_error "notices generation accepted cargo-about 0.8.0"
        exit 1
    }
    case "$unpinned_output" in
        *"requires cargo-about ${pinned_cargo_about_version}"*) ;;
        *)
            print_error "unpinned cargo-about did not name the required version: ${unpinned_output}"
            exit 1
            ;;
    esac
    printf '%s\n' '[rust-dependency-notices-contract] case=reject-unpinned-cargo-about status=success'

    printf '%s\n' '[rust-dependency-notices-contract] case=ci-installs-only-the-pinned-version status=start'
    workflow_path="${repository_root}/.github/workflows/ci.yml"
    installer_path="${repository_root}/scripts/install-verification-tools.sh"
    grep -F 'scripts/install-verification-tools.sh' "$workflow_path" >/dev/null || {
        print_error "CI does not install verification tools through the pinned-archive installer"
        exit 1
    }
    grep -F 'third-party/cargo-about-version' "$installer_path" >/dev/null || {
        print_error "CI installer does not read third-party/cargo-about-version"
        exit 1
    }
    grep -F 'Astronomical requires cargo-about' "$installer_path" >/dev/null || {
        print_error "CI does not fail closed when installed cargo-about drifts"
        exit 1
    }
    if grep -E 'brew install|cargo install' "$installer_path" >/dev/null; then
        print_error "CI installer still compiles or brews verification tools"
        exit 1
    fi
    printf '%s\n' '[rust-dependency-notices-contract] case=ci-installs-only-the-pinned-version status=success'
    printf '%s\n' '[verification-tools-contract] status=start'
    "${repository_root}/scripts/test-install-verification-tools.sh"
    printf '%s\n' '[rust-dependency-notices-contract] status=success'
}

main "$@"
