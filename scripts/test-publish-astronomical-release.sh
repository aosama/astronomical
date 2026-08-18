#!/usr/bin/env sh

# Exercises the release journey with local command doubles so no GitHub state,
# Keychain signing identity, or production app bundle is touched.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=30
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -z "${SANDBOX_DIRECTORY:-}" ] || [ ! -d "$SANDBOX_DIRECTORY" ]; then
        return 0
    fi
    case "$SANDBOX_DIRECTORY" in
        /|.|..) print_error "refusing to remove unsafe release test sandbox" ;;
        *) rm -rf "$SANDBOX_DIRECTORY" ;;
    esac
}
trap cleanup 0

write_fixture_commands() {
    fake_command_directory="$1"

    cat > "${fake_command_directory}/cargo" <<'CARGO'
#!/usr/bin/env sh
printf '%s\n' '{"packages":[{"name":"astronomical-supervisor","version":"0.2.1","repository":"https://github.com/example/astronomical"}]}'
CARGO
    cat > "${fake_command_directory}/jq" <<'JQ'
#!/usr/bin/env sh
case "$*" in
    *'.version'*) printf '%s\n' '0.2.1' ;;
    *'.repository'*) printf '%s\n' 'https://github.com/example/astronomical' ;;
    *'.assets[0].name'*) printf '%s\n' 'Astronomical-0.2.1-macOS-arm64.dmg' ;;
    *) exit 1 ;;
esac
JQ
    cat > "${fake_command_directory}/git" <<'GIT'
#!/usr/bin/env sh
case "${1:-}" in
    status) exit 0 ;;
    rev-parse) printf '%s\n' '0123456789abcdef' ;;
    *) exit 1 ;;
esac
GIT
    cat > "${fake_command_directory}/plutil" <<'PLUTIL'
#!/usr/bin/env sh
case "${2:-}" in
    CFBundleShortVersionString) printf '%s\n' '0.2.1' ;;
    SUFeedURL) printf '%s\n' 'https://example.github.io/astronomical/appcast.xml' ;;
    *) exit 1 ;;
esac
PLUTIL
    cat > "${fake_command_directory}/hdiutil" <<'HDIUTIL'
#!/usr/bin/env sh
for command_argument in "$@"; do output_path="$command_argument"; done
mkdir -p "$(dirname -- "$output_path")"
printf '%s\n' 'fixture dmg' > "$output_path"
HDIUTIL
    cat > "${fake_command_directory}/xmllint" <<'XMLLINT'
#!/usr/bin/env sh
exit 0
XMLLINT
    cat > "${fake_command_directory}/gh" <<'GH'
#!/usr/bin/env sh
printf '%s\n' "$*" >> "${FAKE_GH_LOG:?}"
case "${1:-} ${2:-}" in
    'release view')
        case "$*" in
            *'--json assets'*) [ -f "${FAKE_GH_STATE:?}" ] || exit 1; printf '%s\n' 'Astronomical-0.2.1-macOS-arm64.dmg' ;;
            *) [ -f "${FAKE_GH_STATE:?}" ] ;;
        esac
        ;;
    'release create')
        [ "${FAKE_GH_FAIL_CREATE:-false}" != "true" ] || exit 1
        : > "${FAKE_GH_STATE:?}"
        ;;
    'release edit')
        [ -f "${FAKE_GH_STATE:?}" ]
        ;;
    *) exit 1 ;;
esac
GH
    chmod +x "${fake_command_directory}"/*
}

write_fixture_repository() {
    sandbox_repository="$1"
    repository_root="$2"
    mkdir -p \
        "${sandbox_repository}/scripts" \
        "${sandbox_repository}/site" \
        "${sandbox_repository}/target" \
        "${sandbox_repository}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin"
    cp "${repository_root}/scripts/publish-astronomical-release.sh" \
        "${sandbox_repository}/scripts/publish-astronomical-release.sh"
    chmod +x "${sandbox_repository}/scripts/publish-astronomical-release.sh"
    printf '%s\n' '# Release notes' 'Safe signed update fixture.' > "${sandbox_repository}/release-notes.md"

    cat > "${sandbox_repository}/scripts/make-astronomical-app.sh" <<'BUILDER'
#!/usr/bin/env sh
set -eu
[ "${ASTRONOMICAL_UPDATE_FEED_URL:-}" = 'https://example.github.io/astronomical/appcast.xml' ]
repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
app_bundle="${repository_root}/target/astronomical-macos-stable.noindex/Astronomical.app"
mkdir -p "${app_bundle}/Contents"
printf '%s\n' '<plist><dict></dict></plist>' > "${app_bundle}/Contents/Info.plist"
BUILDER
    chmod +x "${sandbox_repository}/scripts/make-astronomical-app.sh"

    cat > "${sandbox_repository}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin/generate_appcast" <<'APPCAST'
#!/usr/bin/env sh
set -eu
for command_argument in "$@"; do archives_directory="$command_argument"; done
cat > "${archives_directory}/appcast.xml" <<'XML'
<?xml version="1.0"?>
<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
<channel><item><enclosure url="https://github.com/example/astronomical/releases/download/v0.2.1/Astronomical-0.2.1-macOS-arm64.dmg" sparkle:edSignature="fixture" /></item></channel>
</rss>
<!-- sparkle-signatures:
edSignature: fixture
length: 1
-->
XML
APPCAST
    chmod +x "${sandbox_repository}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin/generate_appcast"
}

create_case_sandbox() {
    repository_root="$1"
    case_name="$2"
    case_directory="${SANDBOX_DIRECTORY}/${case_name}"
    sandbox_repository="${case_directory}/repository"
    fake_command_directory="${case_directory}/fake-bin"
    mkdir -p "$sandbox_repository" "$fake_command_directory"
    write_fixture_repository "$sandbox_repository" "$repository_root"
    write_fixture_commands "$fake_command_directory"
    printf '%s\n' "$case_directory"
}

run_prepare_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" prepare)"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' '[release-publisher-test] case=prepare-without-github-mutation status=start'
    (
        CDPATH='' cd -- "$sandbox_repository"
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/publish-astronomical-release.sh \
            --tag v0.2.1 \
            --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/prepared-release"
    )
    [ -s "${sandbox_repository}/prepared-release/Astronomical-0.2.1-macOS-arm64.dmg" ]
    [ -s "${sandbox_repository}/prepared-release/appcast.xml" ]
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    [ ! -e "${case_directory}/gh.log" ]
    printf '%s\n' '[release-publisher-test] case=prepare-without-github-mutation status=success'
}

run_publish_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" publish)"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' '[release-publisher-test] case=publish-asset-before-staging-feed status=start'
    (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" \
        FAKE_GH_STATE="${case_directory}/gh.state" \
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/publish-astronomical-release.sh \
            --tag v0.2.1 \
            --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/published-release" \
            --publish
    )
    [ -s "${sandbox_repository}/site/appcast.xml" ]
    grep -F 'release create v0.2.1' "${case_directory}/gh.log" >/dev/null
    grep -F 'release edit v0.2.1' "${case_directory}/gh.log" >/dev/null
    printf '%s\n' '[release-publisher-test] case=publish-asset-before-staging-feed status=success'
}

run_failed_publish_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" failed-publish)"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' '[release-publisher-test] case=failed-upload-keeps-feed-unchanged status=start'
    if (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_FAIL_CREATE="true" \
        FAKE_GH_LOG="${case_directory}/gh.log" \
        FAKE_GH_STATE="${case_directory}/gh.state" \
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/publish-astronomical-release.sh \
            --tag v0.2.1 \
            --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/failed-release" \
            --publish
    ); then
        print_error "release publishing unexpectedly succeeded"
        exit 1
    fi
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=failed-upload-keeps-feed-unchanged status=success'
}

main() {
    for required_command in grep mktemp timeout; do
        command -v "$required_command" >/dev/null 2>&1 || { print_error "required command is unavailable: $required_command"; exit 2; }
    done
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-release-publisher.XXXXXX")"
    run_prepare_case "$repository_root"
    run_publish_case "$repository_root"
    run_failed_publish_case "$repository_root"
}

main "$@"
