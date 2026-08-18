#!/usr/bin/env sh

# Prepares a signed macOS update locally and optionally publishes its immutable
# GitHub Release asset before staging the signed GitHub Pages appcast.

set -eu

RELEASE_TAG=""
RELEASE_NOTES_FILE=""
OUTPUT_DIRECTORY=""
SHOULD_PUBLISH="false"
STAGING_DIRECTORY=""
CURRENT_STEP=""
CURRENT_STEP_STARTED_AT=0

print_usage() {
    printf '%s\n' "Usage: scripts/publish-astronomical-release.sh --tag vX.Y.Z --notes-file PATH [--output-directory PATH] [--publish]"
    printf '%s\n' ""
    printf '%s\n' "Without --publish, prepares and signs the DMG and appcast locally."
    printf '%s\n' "With --publish, creates and validates a draft GitHub Release, publishes it,"
    printf '%s\n' "then stages site/appcast.xml for review and a later main-branch deployment."
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        print_error "required command is unavailable: $1"
        exit 1
    }
}

start_step() {
    CURRENT_STEP="$1"
    CURRENT_STEP_STARTED_AT="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$CURRENT_STEP"
}

finish_step() {
    printf '%s step=%s status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$CURRENT_STEP" \
        "$(( $(date +%s) - CURRENT_STEP_STARTED_AT ))"
}

cleanup() {
    if [ -z "${STAGING_DIRECTORY:-}" ] || [ ! -d "$STAGING_DIRECTORY" ]; then
        return 0
    fi
    case "$STAGING_DIRECTORY" in
        */target/astronomical-release-staging.*) rm -rf "$STAGING_DIRECTORY" ;;
        *) print_error "refusing to remove unexpected staging directory: $STAGING_DIRECTORY" ;;
    esac
}
trap cleanup 0

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --tag)
                [ "$#" -ge 2 ] || { print_error "--tag requires a value"; exit 2; }
                RELEASE_TAG="$2"
                shift 2
                ;;
            --notes-file)
                [ "$#" -ge 2 ] || { print_error "--notes-file requires a path"; exit 2; }
                RELEASE_NOTES_FILE="$2"
                shift 2
                ;;
            --output-directory)
                [ "$#" -ge 2 ] || { print_error "--output-directory requires a path"; exit 2; }
                OUTPUT_DIRECTORY="$2"
                shift 2
                ;;
            --publish)
                SHOULD_PUBLISH="true"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                print_usage >&2
                exit 2
                ;;
        esac
    done
}

validate_release_identity() {
    repository_root="$1"
    [ -n "$RELEASE_TAG" ] || { print_error "--tag is required"; exit 2; }
    [ -n "$RELEASE_NOTES_FILE" ] || { print_error "--notes-file is required"; exit 2; }
    [ -s "$RELEASE_NOTES_FILE" ] || { print_error "release notes are unavailable or empty: $RELEASE_NOTES_FILE"; exit 1; }
    case "$RELEASE_TAG" in v*) ;; *) print_error "release tag must use vMAJOR.MINOR.PATCH"; exit 2 ;; esac
    release_version="${RELEASE_TAG#v}"
    release_major_version="${release_version%%.*}"
    release_minor_and_patch="${release_version#*.}"
    release_minor_version="${release_minor_and_patch%%.*}"
    release_patch_version="${release_minor_and_patch#*.}"
    if [ "$release_minor_and_patch" = "$release_version" ] \
        || [ "$release_patch_version" = "$release_minor_and_patch" ]; then
        print_error "release tag must use vMAJOR.MINOR.PATCH"
        exit 2
    fi
    for semantic_version_component in "$release_major_version" "$release_minor_version" "$release_patch_version"; do
        case "$semantic_version_component" in
            ''|*[!0-9]*) print_error "release tag must use numeric MAJOR.MINOR.PATCH components"; exit 2 ;;
        esac
    done

    workspace_metadata="$(cargo metadata --no-deps --format-version 1)"
    workspace_version="$(printf '%s' "$workspace_metadata" | jq --raw-output '.packages[] | select(.name == "astronomical-supervisor") | .version')"
    repository_url="$(printf '%s' "$workspace_metadata" | jq --raw-output '.packages[] | select(.name == "astronomical-supervisor") | .repository')"
    [ "$release_version" = "$workspace_version" ] || {
        print_error "release tag ${RELEASE_TAG} does not match workspace version ${workspace_version}"
        exit 1
    }
    case "$repository_url" in
        https://github.com/*/*) ;;
        *) print_error "workspace repository must be an HTTPS GitHub URL"; exit 1 ;;
    esac
    repository_slug="${repository_url#https://github.com/}"
    repository_slug="${repository_slug%.git}"
    repository_owner="${repository_slug%%/*}"
    repository_name="${repository_slug#*/}"
    update_feed_url="https://${repository_owner}.github.io/${repository_name}/appcast.xml"

    [ -z "$(git status --porcelain --untracked-files=normal)" ] || {
        print_error "release preparation requires a clean worktree"
        exit 1
    }
    tagged_commit="$(git rev-parse "${RELEASE_TAG}^{commit}")"
    current_commit="$(git rev-parse HEAD)"
    [ "$tagged_commit" = "$current_commit" ] || {
        print_error "release tag ${RELEASE_TAG} must identify the current commit"
        exit 1
    }

    default_output_directory="${repository_root}/target/releases/${RELEASE_TAG}"
    OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-$default_output_directory}"
    [ ! -e "$OUTPUT_DIRECTORY" ] || {
        print_error "release output already exists: $OUTPUT_DIRECTORY"
        exit 1
    }
}

locate_sparkle_appcast_tool() {
    repository_root="$1"
    for appcast_tool in \
        "${repository_root}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin/generate_appcast" \
        "${repository_root}/apps/astronomical-menu/.build/artifacts/sparkle/bin/generate_appcast"
    do
        if [ -x "$appcast_tool" ]; then
            printf '%s\n' "$appcast_tool"
            return
        fi
    done
    print_error "Sparkle generate_appcast is unavailable after the menu build"
    exit 1
}

prepare_release() {
    repository_root="$1"
    output_parent_directory="$(dirname -- "$OUTPUT_DIRECTORY")"
    mkdir -p "$output_parent_directory"
    STAGING_DIRECTORY="$(mktemp -d "${repository_root}/target/astronomical-release-staging.XXXXXX")"

    start_step "build-stable-app"
    ASTRONOMICAL_UPDATE_FEED_URL="$update_feed_url" \
        "${repository_root}/scripts/make-astronomical-app.sh" --channel stable
    finish_step

    stable_app_bundle="${repository_root}/target/astronomical-macos-stable.noindex/Astronomical.app"
    [ -d "$stable_app_bundle" ] || { print_error "stable app bundle was not produced"; exit 1; }
    bundled_version="$(plutil -extract CFBundleShortVersionString raw -o - "${stable_app_bundle}/Contents/Info.plist")"
    bundled_feed_url="$(plutil -extract SUFeedURL raw -o - "${stable_app_bundle}/Contents/Info.plist")"
    [ "$bundled_version" = "$release_version" ] || { print_error "built app version does not match ${RELEASE_TAG}"; exit 1; }
    [ "$bundled_feed_url" = "$update_feed_url" ] || { print_error "built app update feed does not match this repository"; exit 1; }

    release_asset_name="Astronomical-${release_version}-macOS-arm64.dmg"
    release_dmg="${STAGING_DIRECTORY}/${release_asset_name}"
    start_step "create-full-dmg"
    hdiutil create -volname "Astronomical" -srcfolder "$stable_app_bundle" -format UDZO "$release_dmg"
    [ -s "$release_dmg" ] || { print_error "release DMG was not created"; exit 1; }
    finish_step

    release_notes_copy="${STAGING_DIRECTORY}/Astronomical-${release_version}-macOS-arm64.md"
    cp "$RELEASE_NOTES_FILE" "$release_notes_copy"
    appcast_tool="$(locate_sparkle_appcast_tool "$repository_root")"
    release_download_prefix="https://github.com/${repository_slug}/releases/download/${RELEASE_TAG}/"
    start_step "sign-dmg-and-appcast"
    "$appcast_tool" \
        --download-url-prefix "$release_download_prefix" \
        --embed-release-notes \
        --maximum-deltas 0 \
        -o "${STAGING_DIRECTORY}/appcast.xml" \
        "$STAGING_DIRECTORY"
    generated_appcast="${STAGING_DIRECTORY}/appcast.xml"
    [ -s "$generated_appcast" ] || { print_error "Sparkle did not generate an appcast"; exit 1; }
    xmllint --noout "$generated_appcast"
    grep -F "sparkle:edSignature=" "$generated_appcast" >/dev/null || { print_error "appcast archive signature is missing"; exit 1; }
    grep -F "sparkle-signatures:" "$generated_appcast" >/dev/null || { print_error "signed appcast envelope is missing"; exit 1; }
    grep -F "${release_download_prefix}${release_asset_name}" "$generated_appcast" >/dev/null || { print_error "appcast release URL is incorrect"; exit 1; }
    finish_step

    mv "$STAGING_DIRECTORY" "$OUTPUT_DIRECTORY"
    STAGING_DIRECTORY=""
    release_dmg="${OUTPUT_DIRECTORY}/${release_asset_name}"
    generated_appcast="${OUTPUT_DIRECTORY}/appcast.xml"
}

publish_release() {
    repository_root="$1"
    start_step "create-draft-github-release"
    if gh release view "$RELEASE_TAG" --repo "$repository_slug" >/dev/null 2>&1; then
        print_error "GitHub Release already exists for ${RELEASE_TAG}"
        exit 1
    fi
    gh release create "$RELEASE_TAG" "$release_dmg" \
        --repo "$repository_slug" \
        --verify-tag \
        --draft \
        --title "Astronomical ${release_version}" \
        --notes-file "$RELEASE_NOTES_FILE"
    finish_step

    start_step "verify-and-publish-github-release"
    uploaded_asset_name="$(gh release view "$RELEASE_TAG" --repo "$repository_slug" --json assets --jq '.assets[0].name')"
    [ "$uploaded_asset_name" = "$release_asset_name" ] || {
        print_error "draft GitHub Release does not contain the expected DMG"
        exit 1
    }
    gh release edit "$RELEASE_TAG" --repo "$repository_slug" --draft=false
    finish_step

    start_step "stage-signed-pages-appcast"
    cp "$generated_appcast" "${repository_root}/site/appcast.xml"
    finish_step
}

main() {
    parse_arguments "$@"
    for command_name in cargo git grep hdiutil jq mktemp plutil xmllint; do require_command "$command_name"; done
    if [ "$SHOULD_PUBLISH" = "true" ]; then require_command gh; fi
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"

    start_step "validate-release-identity"
    validate_release_identity "$repository_root"
    finish_step
    prepare_release "$repository_root"
    if [ "$SHOULD_PUBLISH" = "true" ]; then publish_release "$repository_root"; fi

    printf '%s\n' "Prepared signed release: ${release_dmg}"
    if [ "$SHOULD_PUBLISH" = "true" ]; then
        printf '%s\n' "Published GitHub Release ${RELEASE_TAG}; review and commit site/appcast.xml to activate the update feed."
    else
        printf '%s\n' "No GitHub state changed. Re-run with --publish after reviewing the local artifacts."
    fi
}

main "$@"
