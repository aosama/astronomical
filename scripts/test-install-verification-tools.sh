#!/usr/bin/env sh

# Proves CI verification tools come from pinned GitHub Release archives, not a
# crates.io compile or Homebrew install.

set -eu

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe verification-tools sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

write_executable() {
    executable_path="$1"
    mkdir -p "$(dirname -- "$executable_path")"
    cat > "$executable_path"
    chmod +x "$executable_path"
}

write_version_binary() {
    executable_path="$1"
    version_line="$2"
    write_executable "$executable_path" <<EOF
#!/usr/bin/env sh
set -eu
if [ "\${1:-}" = "--version" ]; then
    printf '%s\\n' "${version_line}"
    exit 0
fi
exit 0
EOF
}

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

create_release_archive() {
    archive_path="$1"
    binary_name="$2"
    version_line="$3"
    staging_directory="$4"
    payload_directory="${staging_directory}/${binary_name}-payload"
    rm -rf "$payload_directory"
    mkdir -p "$payload_directory"
    write_version_binary "${payload_directory}/${binary_name}" "$version_line"
    tar -czf "$archive_path" -C "$staging_directory" "$(basename -- "$payload_directory")"
    sha256_file "$archive_path" > "${archive_path}.sha256"
}

create_fake_commands() {
    fake_command_directory="$1"
    fixture_directory="$2"
    mkdir -p "$fake_command_directory"
    write_executable "${fake_command_directory}/curl" <<EOF
#!/usr/bin/env sh
set -eu
output_path=""
download_url=""
while [ "\$#" -gt 0 ]; do
    case "\$1" in
        -o)
            output_path="\$2"
            shift 2
            ;;
        --retry|--retry-delay)
            shift 2
            ;;
        -fsSL|-f|-s|-S|-L)
            shift
            ;;
        -*)
            shift
            ;;
        *)
            download_url="\$1"
            shift
            ;;
    esac
done
[ -n "\$output_path" ] && [ -n "\$download_url" ] || exit 1
printf '%s\\n' "\$download_url" >> "${SANDBOX_DIRECTORY}/curl.log"
archive_file=""
case "\$download_url" in
    *"/cargo-about-"*".tar.gz.sha256") archive_file="cargo-about.tar.gz.sha256" ;;
    *"/cargo-about-"*".tar.gz") archive_file="cargo-about.tar.gz" ;;
    *"/sccache-v"*".tar.gz.sha256") archive_file="sccache.tar.gz.sha256" ;;
    *"/sccache-v"*".tar.gz") archive_file="sccache.tar.gz" ;;
    *) exit 1 ;;
esac
cp "${fixture_directory}/\${archive_file}" "\$output_path"
EOF
    write_executable "${fake_command_directory}/brew" <<'BREW'
#!/usr/bin/env sh
printf '%s\n' "Error: brew must not install verification tools" >&2
exit 1
BREW
    write_executable "${fake_command_directory}/cargo" <<'CARGO'
#!/usr/bin/env sh
printf '%s\n' "Error: cargo install must not compile verification tools" >&2
exit 1
CARGO
}

run_installer() {
    PATH="${SANDBOX_DIRECTORY}/fake-bin:${PATH}" \
        HOME="${SANDBOX_DIRECTORY}/home" \
        "$INSTALLER_SCRIPT" --prefix "$INSTALL_PREFIX"
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    INSTALLER_SCRIPT="${repository_root}/scripts/install-verification-tools.sh"
    pinned_cargo_about_version="$(tr -d '[:space:]' < "${repository_root}/third-party/cargo-about-version")"
    pinned_sccache_version="$(tr -d '[:space:]' < "${repository_root}/third-party/sccache-version")"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-verification-tools-test.XXXXXX")"
    fixture_directory="${SANDBOX_DIRECTORY}/fixtures"
    INSTALL_PREFIX="${SANDBOX_DIRECTORY}/prefix"
    mkdir -p "$fixture_directory" "$INSTALL_PREFIX" "${SANDBOX_DIRECTORY}/home"
    create_fake_commands "${SANDBOX_DIRECTORY}/fake-bin" "$fixture_directory"
    create_release_archive \
        "${fixture_directory}/cargo-about.tar.gz" \
        cargo-about \
        "cargo-about ${pinned_cargo_about_version}" \
        "$SANDBOX_DIRECTORY"
    create_release_archive \
        "${fixture_directory}/sccache.tar.gz" \
        sccache \
        "sccache ${pinned_sccache_version}" \
        "$SANDBOX_DIRECTORY"

    printf '%s\n' '[verification-tools-contract] case=skip-matching-prefix status=start'
    write_version_binary "${INSTALL_PREFIX}/cargo-about" "cargo-about ${pinned_cargo_about_version}"
    write_version_binary "${INSTALL_PREFIX}/sccache" "sccache ${pinned_sccache_version}"
    : > "${SANDBOX_DIRECTORY}/curl.log"
    run_installer >/dev/null
    [ ! -s "${SANDBOX_DIRECTORY}/curl.log" ] || {
        print_error "matching prefix still downloaded archives"
        exit 1
    }
    printf '%s\n' '[verification-tools-contract] case=skip-matching-prefix status=success'

    printf '%s\n' '[verification-tools-contract] case=install-pinned-prebuilts status=start'
    rm -f "${INSTALL_PREFIX}/cargo-about" "${INSTALL_PREFIX}/sccache" "${SANDBOX_DIRECTORY}/curl.log"
    install_output="$(run_installer)"
    [ "$("${INSTALL_PREFIX}/cargo-about" --version)" = "cargo-about ${pinned_cargo_about_version}" ] || {
        print_error "installer did not place the pinned cargo-about"
        exit 1
    }
    [ "$("${INSTALL_PREFIX}/sccache" --version)" = "sccache ${pinned_sccache_version}" ] || {
        print_error "installer did not place the pinned sccache"
        exit 1
    }
    grep -F "github.com/EmbarkStudios/cargo-about/releases/download/${pinned_cargo_about_version}/" \
        "${SANDBOX_DIRECTORY}/curl.log" >/dev/null || {
        print_error "cargo-about was not fetched from its pinned GitHub Release"
        exit 1
    }
    grep -F "github.com/mozilla/sccache/releases/download/v${pinned_sccache_version}/" \
        "${SANDBOX_DIRECTORY}/curl.log" >/dev/null || {
        print_error "sccache was not fetched from its pinned GitHub Release"
        exit 1
    }
    case "$install_output" in
        *"[verification-tools] status=success"*) ;;
        *)
            print_error "installer did not report success: ${install_output}"
            exit 1
            ;;
    esac
    printf '%s\n' '[verification-tools-contract] case=install-pinned-prebuilts status=success'

    printf '%s\n' '[verification-tools-contract] case=replace-stale-prefix status=start'
    write_version_binary "${INSTALL_PREFIX}/cargo-about" "cargo-about 0.8.0"
    write_version_binary "${INSTALL_PREFIX}/sccache" "sccache 0.1.0"
    : > "${SANDBOX_DIRECTORY}/curl.log"
    run_installer >/dev/null
    [ "$("${INSTALL_PREFIX}/cargo-about" --version)" = "cargo-about ${pinned_cargo_about_version}" ] || {
        print_error "stale cargo-about was not replaced"
        exit 1
    }
    [ "$("${INSTALL_PREFIX}/sccache" --version)" = "sccache ${pinned_sccache_version}" ] || {
        print_error "stale sccache was not replaced"
        exit 1
    }
    [ -s "${SANDBOX_DIRECTORY}/curl.log" ] || {
        print_error "stale prefix did not download replacements"
        exit 1
    }
    printf '%s\n' '[verification-tools-contract] case=replace-stale-prefix status=success'

    printf '%s\n' '[verification-tools-contract] case=reject-digest-mismatch status=start'
    rm -f "${INSTALL_PREFIX}/cargo-about" "${INSTALL_PREFIX}/sccache"
    printf '%s\n' '0000000000000000000000000000000000000000000000000000000000000000' \
        > "${fixture_directory}/cargo-about.tar.gz.sha256"
    : > "${SANDBOX_DIRECTORY}/curl.log"
    if digest_output="$(run_installer 2>&1)"; then
        print_error "installer accepted a SHA-256 mismatch"
        exit 1
    fi
    case "$digest_output" in
        *"SHA-256 mismatch for cargo-about"*) ;;
        *)
            print_error "digest failure did not name cargo-about: ${digest_output}"
            exit 1
            ;;
    esac
    [ ! -e "${INSTALL_PREFIX}/cargo-about" ] || {
        print_error "digest mismatch still installed cargo-about"
        exit 1
    }
    sha256_file "${fixture_directory}/cargo-about.tar.gz" > "${fixture_directory}/cargo-about.tar.gz.sha256"
    printf '%s\n' '[verification-tools-contract] case=reject-digest-mismatch status=success'

    printf '%s\n' '[verification-tools-contract] case=reject-version-mismatch status=start'
    create_release_archive \
        "${fixture_directory}/cargo-about.tar.gz" \
        cargo-about \
        "cargo-about 0.8.0" \
        "$SANDBOX_DIRECTORY"
    rm -f "${INSTALL_PREFIX}/cargo-about" "${INSTALL_PREFIX}/sccache"
    if version_output="$(run_installer 2>&1)"; then
        print_error "installer accepted cargo-about 0.8.0"
        exit 1
    fi
    case "$version_output" in
        *"Astronomical requires cargo-about"*"${pinned_cargo_about_version}"*) ;;
        *)
            print_error "version mismatch did not fail closed on the pin: ${version_output}"
            exit 1
            ;;
    esac
    printf '%s\n' '[verification-tools-contract] case=reject-version-mismatch status=success'
    printf '%s\n' '[verification-tools-contract] status=success'
}

main "$@"
