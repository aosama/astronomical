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
    cat > "${fake_command_directory}/security" <<'SECURITY'
#!/usr/bin/env sh
printf '%s\n' '  1) FIXTURE "Developer ID Application: Example (ABCDE12345)"'
SECURITY
    cat > "${fake_command_directory}/xcrun" <<'XCRUN'
#!/usr/bin/env sh
[ "${1:-} ${2:-}" = "notarytool history" ]
XCRUN
    cat > "${fake_command_directory}/xmllint" <<'XMLLINT'
#!/usr/bin/env sh
exit 0
XMLLINT
    cat > "${fake_command_directory}/gh" <<'GH'
#!/usr/bin/env sh
printf '%s\n' "$*" >> "${FAKE_GH_LOG:?}"
case "${1:-} ${2:-}" in
    'api repos/example/astronomical/git/ref/tags/v0.2.1')
        printf '{"object":{"type":"commit","sha":"%s"}}\n' "${FAKE_REMOTE_COMMIT:-0123456789abcdef}"
        ;;
    'release view')
        case "$*" in
            *'--json assets'*) [ -f "${FAKE_GH_STATE:?}" ] || exit 1; cat "${FAKE_GH_STATE}" ;;
            *) [ -f "${FAKE_GH_STATE:?}" ] ;;
        esac
        ;;
    'release create')
        [ "${FAKE_GH_FAIL_CREATE:-false}" != "true" ] || exit 1
        for command_argument in "$@"; do
            case "$command_argument" in *.dmg) release_dmg="$command_argument" ;; esac
        done
        digest="$(shasum -a 256 "${release_dmg:?}")"
        digest="${digest%% *}"
        [ "${FAKE_GH_BAD_DIGEST:-false}" != "true" ] || digest="incorrect"
        size="$(stat -f '%z' "$release_dmg")"
        printf '{"tagName":"v0.2.1","name":"Astronomical 0.2.1","body":"# Release notes\\nSafe signed update fixture.","isDraft":true,"isImmutable":false,"isPrerelease":false,"assets":[{"name":"Astronomical-0.2.1-macOS-arm64.dmg","digest":"sha256:%s","size":%s,"state":"uploaded"}]}\n' \
            "$digest" "$size" > "${FAKE_GH_STATE:?}"
        ;;
    'release upload')
        for command_argument in "$@"; do
            case "$command_argument" in *.dmg) release_dmg="$command_argument" ;; esac
        done
        digest="$(shasum -a 256 "${release_dmg:?}")"
        digest="${digest%% *}"
        size="$(stat -f '%z' "$release_dmg")"
        jq --arg digest "sha256:${digest}" --argjson size "$size" \
            '.assets = [{name:"Astronomical-0.2.1-macOS-arm64.dmg",digest:$digest,size:$size,state:"uploaded"}]' \
            "${FAKE_GH_STATE:?}" > "${FAKE_GH_STATE}.next"
        mv "${FAKE_GH_STATE}.next" "$FAKE_GH_STATE"
        ;;
    'release edit')
        [ -f "${FAKE_GH_STATE:?}" ]
        jq '.isDraft = false' "$FAKE_GH_STATE" > "${FAKE_GH_STATE}.next"
        mv "${FAKE_GH_STATE}.next" "$FAKE_GH_STATE"
        ;;
    *) exit 1 ;;
esac
GH
    cat > "${fake_command_directory}/install" <<'INSTALL'
#!/usr/bin/env sh
printf '%s\n' 'stage appcast' >> "${FAKE_GH_LOG:?}"
/usr/bin/install "$@"
INSTALL
    chmod +x "${fake_command_directory}"/*
}

write_fixture_repository() {
    sandbox_repository="$1"
    repository_root="$2"
    mkdir -p \
        "${sandbox_repository}/scripts/release" \
        "${sandbox_repository}/site" \
        "${sandbox_repository}/target" \
        "${sandbox_repository}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin"
    cp "${repository_root}/scripts/release/prepare-and-publish.sh" \
        "${sandbox_repository}/scripts/release/prepare-and-publish.sh"
    chmod +x "${sandbox_repository}/scripts/release/prepare-and-publish.sh"
    printf '%s\n' '# Release notes' 'Safe signed update fixture.' > "${sandbox_repository}/release-notes.md"

    cat > "${sandbox_repository}/scripts/release/build-stable-app.sh" <<'BUILDER'
#!/usr/bin/env sh
set -eu
[ "${ASTRONOMICAL_UPDATE_FEED_URL:-}" = 'https://example.github.io/astronomical/appcast.xml' ]
case "$*" in *'--signing-identity Developer ID Application: Example (ABCDE12345)'*) ;; *) exit 1 ;; esac
repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
app_bundle="${repository_root}/target/astronomical-macos-stable.noindex/Astronomical.app"
mkdir -p "${app_bundle}/Contents"
printf '%s\n' '<plist><dict></dict></plist>' > "${app_bundle}/Contents/Info.plist"
BUILDER
    chmod +x "${sandbox_repository}/scripts/release/build-stable-app.sh"

    cat > "${sandbox_repository}/scripts/release/validate-distribution-app.sh" <<'VALIDATE_APP'
#!/usr/bin/env sh
set -eu
case "$*" in *'--team-id ABCDE12345'*) ;; *) exit 1 ;; esac
VALIDATE_APP
    cat > "${sandbox_repository}/scripts/release/create-dmg.sh" <<'MAKE_DMG'
#!/usr/bin/env sh
set -eu
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then output_path="$2"; break; fi
    shift
done
printf '%s\n' 'fixture notarized dmg' > "${output_path:?}"
MAKE_DMG
    cat > "${sandbox_repository}/scripts/release/notarize-dmg.sh" <<'NOTARIZE'
#!/usr/bin/env sh
set -eu
case "$*" in *'--notary-profile Fixture Notarization'*) ;; *) exit 1 ;; esac
[ "${FAKE_NOTARIZATION_REJECTED:-false}" != "true" ]
NOTARIZE
    cat > "${sandbox_repository}/scripts/release/validate-dmg.sh" <<'VALIDATE_DMG'
#!/usr/bin/env sh
exit 0
VALIDATE_DMG
    chmod +x \
        "${sandbox_repository}/scripts/release/validate-distribution-app.sh" \
        "${sandbox_repository}/scripts/release/create-dmg.sh" \
        "${sandbox_repository}/scripts/release/notarize-dmg.sh" \
        "${sandbox_repository}/scripts/release/validate-dmg.sh"

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
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 \
            --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/prepared-release" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 \
            --notary-profile "Fixture Notarization"
    )
    [ -s "${sandbox_repository}/prepared-release/Astronomical-0.2.1-macOS-arm64.dmg" ]
    [ -s "${sandbox_repository}/prepared-release/appcast.xml" ]
    [ -s "${sandbox_repository}/prepared-release/release-manifest.json" ]
    [ -s "${sandbox_repository}/prepared-release/Astronomical-0.2.1-macOS-arm64.md" ]
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
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/published-release" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 --notary-profile "Fixture Notarization"
    )
    (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" \
        FAKE_GH_STATE="${case_directory}/gh.state" \
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 \
            --output-directory "${sandbox_repository}/published-release" \
            --publish
    )
    [ -s "${sandbox_repository}/site/appcast.xml" ]
    grep -F 'release create v0.2.1' "${case_directory}/gh.log" >/dev/null
    grep -F 'release edit v0.2.1' "${case_directory}/gh.log" >/dev/null
    edit_line="$(grep -n -F 'release edit v0.2.1' "${case_directory}/gh.log")"
    edit_line="${edit_line%%:*}"
    stage_line="$(grep -n -F 'stage appcast' "${case_directory}/gh.log")"
    stage_line="${stage_line%%:*}"
    [ "$edit_line" -lt "$stage_line" ] || { print_error "appcast was staged before the release became public"; exit 1; }
    (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" \
        FAKE_GH_STATE="${case_directory}/gh.state" \
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 \
            --output-directory "${sandbox_repository}/published-release" \
            --publish
    )
    [ "$(grep -c -F 'release create v0.2.1' "${case_directory}/gh.log")" = "1" ] || {
        print_error "resume unexpectedly recreated the GitHub Release"
        exit 1
    }
    [ "$(grep -c -F 'release edit v0.2.1' "${case_directory}/gh.log")" = "1" ] || {
        print_error "resume unexpectedly republished the GitHub Release"
        exit 1
    }
    printf '%s\n' '[release-publisher-test] case=publish-asset-before-staging-feed status=success'
}

run_prepared_artifact_tamper_case() {
    repository_root="$1"
    artifact_relative_path="$2"
    case_name="$3"
    case_directory="$(create_case_sandbox "$repository_root" "$case_name")"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' "[release-publisher-test] case=${case_name} status=start"
    (
        CDPATH='' cd -- "$sandbox_repository"
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/tampered-release" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 --notary-profile "Fixture Notarization"
    )
    printf '%s\n' 'tampered after preparation' >> "${sandbox_repository}/tampered-release/${artifact_relative_path}"
    if (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" FAKE_GH_STATE="${case_directory}/gh.state" \
            PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --output-directory "${sandbox_repository}/tampered-release" --publish
    ); then
        print_error "publisher unexpectedly accepted tampered ${artifact_relative_path}"
        exit 1
    fi
    [ ! -e "${case_directory}/gh.log" ] || { print_error "tampered artifact unexpectedly reached GitHub"; exit 1; }
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' "[release-publisher-test] case=${case_name} status=success"
}

run_interrupted_draft_recovery_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" interrupted-draft)"
    sandbox_repository="${case_directory}/repository"
    prepared_directory="${sandbox_repository}/interrupted-release"
    printf '%s\n' '[release-publisher-test] case=interrupted-draft-is-resumed status=start'
    (
        CDPATH='' cd -- "$sandbox_repository"
        PATH="${case_directory}/fake-bin:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "$prepared_directory" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 --notary-profile "Fixture Notarization"
    )
    printf '%s\n' '{"tagName":"v0.2.1","name":"Astronomical 0.2.1","body":"# Release notes\nSafe signed update fixture.","isDraft":true,"isImmutable":false,"isPrerelease":false,"assets":[]}' \
        > "${case_directory}/gh.state"
    (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" FAKE_GH_STATE="${case_directory}/gh.state" \
            PATH="${case_directory}/fake-bin:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --output-directory "$prepared_directory" --publish
    )
    grep -F 'release upload v0.2.1' "${case_directory}/gh.log" >/dev/null || {
        print_error "interrupted draft did not resume its asset upload"
        exit 1
    }
    [ -s "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=interrupted-draft-is-resumed status=success'

    printf '%s\n' '[release-publisher-test] case=prerelease-cannot-activate-stable-feed status=start'
    rm "${sandbox_repository}/site/appcast.xml"
    jq '.isPrerelease = true' "${case_directory}/gh.state" > "${case_directory}/gh.state.next"
    mv "${case_directory}/gh.state.next" "${case_directory}/gh.state"
    if (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_LOG="${case_directory}/gh.log" FAKE_GH_STATE="${case_directory}/gh.state" \
            PATH="${case_directory}/fake-bin:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --output-directory "$prepared_directory" --publish
    ); then
        print_error "Stable publisher accepted a prerelease"
        exit 1
    fi
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=prerelease-cannot-activate-stable-feed status=success'

    printf '%s\n' '[release-publisher-test] case=remote-tag-must-match-prepared-commit status=start'
    if (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_REMOTE_COMMIT=ffffffffffffffff FAKE_GH_LOG="${case_directory}/gh.log" \
            FAKE_GH_STATE="${case_directory}/gh.state" PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --output-directory "$prepared_directory" --publish
    ); then
        print_error "publisher accepted a remote tag for another commit"
        exit 1
    fi
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=remote-tag-must-match-prepared-commit status=success'
}

run_failed_publish_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" failed-publish)"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' '[release-publisher-test] case=failed-upload-keeps-feed-unchanged status=start'
    (
        CDPATH='' cd -- "$sandbox_repository"
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/failed-release" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 --notary-profile "Fixture Notarization"
    )
    if (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_FAIL_CREATE="true" \
        FAKE_GH_LOG="${case_directory}/gh.log" \
        FAKE_GH_STATE="${case_directory}/gh.state" \
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" \
            scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 \
            --output-directory "${sandbox_repository}/failed-release" \
            --publish
    ); then
        print_error "release publishing unexpectedly succeeded"
        exit 1
    fi
    [ ! -e "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=failed-upload-keeps-feed-unchanged status=success'
}

run_digest_mismatch_recovery_case() {
    repository_root="$1"
    case_directory="$(create_case_sandbox "$repository_root" digest-mismatch)"
    sandbox_repository="${case_directory}/repository"
    printf '%s\n' '[release-publisher-test] case=digest-mismatch-is-repaired-before-publication status=start'
    (
        CDPATH='' cd -- "$sandbox_repository"
        PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 --notes-file "${sandbox_repository}/release-notes.md" \
            --output-directory "${sandbox_repository}/digest-release" \
            --signing-identity "Developer ID Application: Example (ABCDE12345)" \
            --team-id ABCDE12345 --notary-profile "Fixture Notarization"
    )
    (
        CDPATH='' cd -- "$sandbox_repository"
        FAKE_GH_BAD_DIGEST=true FAKE_GH_LOG="${case_directory}/gh.log" \
            FAKE_GH_STATE="${case_directory}/gh.state" PATH="${case_directory}/fake-bin:${PATH}" \
            timeout "$SUBJECT_TIMEOUT_SECONDS" scripts/release/prepare-and-publish.sh \
            --tag v0.2.1 \
            --output-directory "${sandbox_repository}/digest-release" --publish
    )
    grep -F 'release upload v0.2.1' "${case_directory}/gh.log" | grep -F -- '--clobber' >/dev/null || {
        print_error "digest mismatch was not replaced from the prepared artifact"
        exit 1
    }
    [ -s "${sandbox_repository}/site/appcast.xml" ]
    printf '%s\n' '[release-publisher-test] case=digest-mismatch-is-repaired-before-publication status=success'
}

main() {
    for required_command in grep mktemp timeout; do
        command -v "$required_command" >/dev/null 2>&1 || { print_error "required command is unavailable: $required_command"; exit 2; }
    done
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-release-publisher.XXXXXX")"
    run_prepare_case "$repository_root"
    run_publish_case "$repository_root"
    run_interrupted_draft_recovery_case "$repository_root"
    run_prepared_artifact_tamper_case "$repository_root" appcast.xml tampered-appcast-is-rejected
    run_prepared_artifact_tamper_case "$repository_root" Astronomical-0.2.1-macOS-arm64.md tampered-notes-are-rejected
    run_failed_publish_case "$repository_root"
    run_digest_mismatch_recovery_case "$repository_root"
}

main "$@"
