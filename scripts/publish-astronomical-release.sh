#!/usr/bin/env sh

# Prepares a signed macOS update locally and optionally publishes its immutable
# GitHub Release asset before staging the signed GitHub Pages appcast.

set -eu

RELEASE_TAG=""
RELEASE_NOTES_FILE=""
OUTPUT_DIRECTORY=""
SHOULD_PUBLISH="false"
SIGNING_IDENTITY=""
EXPECTED_TEAM_ID=""
NOTARY_PROFILE=""
STAGING_DIRECTORY=""
CURRENT_STEP=""
CURRENT_STEP_STARTED_AT=0

print_usage() {
    printf '%s\n' "Usage: scripts/publish-astronomical-release.sh --tag vX.Y.Z --notes-file PATH [--output-directory PATH]"
    printf '%s\n' "       --signing-identity NAME --team-id ID --notary-profile NAME"
    printf '%s\n' "       scripts/publish-astronomical-release.sh --tag vX.Y.Z"
    printf '%s\n' "       --output-directory PATH --publish"
    printf '%s\n' ""
    printf '%s\n' "Without --publish, prepares and signs the DMG and appcast locally."
    printf '%s\n' "With --publish, uploads the exact previously prepared directory, publishes it,"
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
            --signing-identity)
                [ "$#" -ge 2 ] || { print_error "--signing-identity requires a value"; exit 2; }
                SIGNING_IDENTITY="$2"
                shift 2
                ;;
            --team-id)
                [ "$#" -ge 2 ] || { print_error "--team-id requires a value"; exit 2; }
                EXPECTED_TEAM_ID="$2"
                shift 2
                ;;
            --notary-profile)
                [ "$#" -ge 2 ] || { print_error "--notary-profile requires a value"; exit 2; }
                NOTARY_PROFILE="$2"
                shift 2
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
    if [ "$SHOULD_PUBLISH" = "false" ]; then
        [ -n "$RELEASE_NOTES_FILE" ] || { print_error "--notes-file is required while preparing a release"; exit 2; }
        [ -s "$RELEASE_NOTES_FILE" ] || { print_error "release notes are unavailable or empty: $RELEASE_NOTES_FILE"; exit 1; }
    fi
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
    if [ "$SHOULD_PUBLISH" = "true" ]; then
        [ -d "$OUTPUT_DIRECTORY" ] || { print_error "prepared release directory is unavailable: ${OUTPUT_DIRECTORY}"; exit 1; }
    else
        [ ! -e "$OUTPUT_DIRECTORY" ] || { print_error "release output already exists: ${OUTPUT_DIRECTORY}"; exit 1; }
    fi
}

validate_distribution_credentials() {
    repository_root="$1"
    [ -n "$SIGNING_IDENTITY" ] || { print_error "--signing-identity is required while preparing a release"; exit 2; }
    case "$EXPECTED_TEAM_ID" in ???????*) ;; *) print_error "--team-id is required while preparing a release"; exit 2 ;; esac
    [ -n "$NOTARY_PROFILE" ] || { print_error "--notary-profile is required while preparing a release"; exit 2; }
    security find-identity -v -p codesigning | grep -F "\"${SIGNING_IDENTITY}\"" >/dev/null || {
        print_error "configured Developer ID Application identity is unavailable"
        exit 1
    }
    case "$SIGNING_IDENTITY" in
        "Developer ID Application:"*) ;;
        *) print_error "configured identity must be a Developer ID Application identity"; exit 1 ;;
    esac
    xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null
    for helper_script in \
        make-astronomical-dmg.sh \
        notarize-astronomical-dmg.sh \
        validate-astronomical-distribution-app.sh \
        validate-astronomical-dmg.sh
    do
        [ -x "${repository_root}/scripts/${helper_script}" ] || {
            print_error "release helper is unavailable: scripts/${helper_script}"
            exit 1
        }
    done
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
        "${repository_root}/scripts/make-astronomical-app.sh" --channel stable \
            --signing-identity "$SIGNING_IDENTITY"
    finish_step

    stable_app_bundle="${repository_root}/target/astronomical-macos-stable.noindex/Astronomical.app"
    [ -d "$stable_app_bundle" ] || { print_error "stable app bundle was not produced"; exit 1; }
    bundled_version="$(plutil -extract CFBundleShortVersionString raw -o - "${stable_app_bundle}/Contents/Info.plist")"
    bundled_feed_url="$(plutil -extract SUFeedURL raw -o - "${stable_app_bundle}/Contents/Info.plist")"
    [ "$bundled_version" = "$release_version" ] || { print_error "built app version does not match ${RELEASE_TAG}"; exit 1; }
    [ "$bundled_feed_url" = "$update_feed_url" ] || { print_error "built app update feed does not match this repository"; exit 1; }

    start_step "validate-developer-id-app"
    "${repository_root}/scripts/validate-astronomical-distribution-app.sh" \
        --app-bundle "$stable_app_bundle" --team-id "$EXPECTED_TEAM_ID"
    finish_step

    release_asset_name="Astronomical-${release_version}-macOS-arm64.dmg"
    release_dmg="${STAGING_DIRECTORY}/${release_asset_name}"
    start_step "create-drag-to-applications-dmg"
    "${repository_root}/scripts/make-astronomical-dmg.sh" \
        --app-bundle "$stable_app_bundle" --output "$release_dmg"
    finish_step

    start_step "notarize-and-staple-dmg"
    "${repository_root}/scripts/notarize-astronomical-dmg.sh" \
        --dmg "$release_dmg" --signing-identity "$SIGNING_IDENTITY" \
        --notary-profile "$NOTARY_PROFILE"
    finish_step

    start_step "validate-notarized-dmg"
    "${repository_root}/scripts/validate-astronomical-dmg.sh" --dmg "$release_dmg"
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

    release_dmg_sha256="$(shasum -a 256 "$release_dmg")"
    release_dmg_sha256="${release_dmg_sha256%% *}"
    release_dmg_size="$(stat -f '%z' "$release_dmg")"
    release_notes_name="$(basename -- "$release_notes_copy")"
    release_notes_sha256="$(shasum -a 256 "$release_notes_copy")"
    release_notes_sha256="${release_notes_sha256%% *}"
    release_notes_size="$(stat -f '%z' "$release_notes_copy")"
    appcast_sha256="$(shasum -a 256 "$generated_appcast")"
    appcast_sha256="${appcast_sha256%% *}"
    appcast_size="$(stat -f '%z' "$generated_appcast")"
    jq --null-input \
        --arg tag "$RELEASE_TAG" \
        --arg commit "$current_commit" \
        --arg version "$release_version" \
        --arg asset_name "$release_asset_name" \
        --arg sha256 "$release_dmg_sha256" \
        --argjson size "$release_dmg_size" \
        --arg notes_name "$release_notes_name" \
        --arg notes_sha256 "$release_notes_sha256" \
        --argjson notes_size "$release_notes_size" \
        --arg appcast_sha256 "$appcast_sha256" \
        --argjson appcast_size "$appcast_size" \
        '{tag:$tag,commit:$commit,version:$version,
          dmg:{name:$asset_name,sha256:$sha256,size:$size},
          notes:{name:$notes_name,sha256:$notes_sha256,size:$notes_size},
          appcast:{name:"appcast.xml",sha256:$appcast_sha256,size:$appcast_size}}' \
        > "${STAGING_DIRECTORY}/release-manifest.json"

    mv "$STAGING_DIRECTORY" "$OUTPUT_DIRECTORY"
    STAGING_DIRECTORY=""
    release_dmg="${OUTPUT_DIRECTORY}/${release_asset_name}"
    generated_appcast="${OUTPUT_DIRECTORY}/appcast.xml"
}

load_prepared_release() {
    manifest_path="${OUTPUT_DIRECTORY}/release-manifest.json"
    [ -s "$manifest_path" ] || { print_error "prepared release manifest is unavailable"; exit 1; }
    release_asset_name="$(jq --raw-output '.dmg.name' "$manifest_path")"
    release_dmg="${OUTPUT_DIRECTORY}/${release_asset_name}"
    generated_appcast="${OUTPUT_DIRECTORY}/appcast.xml"
    release_notes_name="$(jq --raw-output '.notes.name' "$manifest_path")"
    release_notes_copy="${OUTPUT_DIRECTORY}/${release_notes_name}"
    prepared_tag="$(jq --raw-output '.tag' "$manifest_path")"
    prepared_commit="$(jq --raw-output '.commit' "$manifest_path")"
    prepared_version="$(jq --raw-output '.version' "$manifest_path")"
    release_dmg_sha256="$(jq --raw-output '.dmg.sha256' "$manifest_path")"
    release_dmg_size="$(jq --raw-output '.dmg.size' "$manifest_path")"
    release_notes_sha256="$(jq --raw-output '.notes.sha256' "$manifest_path")"
    release_notes_size="$(jq --raw-output '.notes.size' "$manifest_path")"
    appcast_sha256="$(jq --raw-output '.appcast.sha256' "$manifest_path")"
    appcast_size="$(jq --raw-output '.appcast.size' "$manifest_path")"
    [ "$prepared_tag" = "$RELEASE_TAG" ] || { print_error "prepared release tag does not match"; exit 1; }
    [ "$prepared_commit" = "$current_commit" ] || { print_error "prepared release commit does not match HEAD"; exit 1; }
    [ "$prepared_version" = "$release_version" ] || { print_error "prepared release version does not match"; exit 1; }
    [ -s "$release_dmg" ] && [ -s "$generated_appcast" ] && [ -s "$release_notes_copy" ] || {
        print_error "prepared release artifacts are incomplete"
        exit 1
    }
    prepared_dmg_sha256="$(shasum -a 256 "$release_dmg")"
    prepared_dmg_sha256="${prepared_dmg_sha256%% *}"
    [ "$prepared_dmg_sha256" = "$release_dmg_sha256" ] || { print_error "prepared DMG digest does not match its manifest"; exit 1; }
    [ "$(stat -f '%z' "$release_dmg")" = "$release_dmg_size" ] || { print_error "prepared DMG size does not match its manifest"; exit 1; }
    prepared_notes_sha256="$(shasum -a 256 "$release_notes_copy")"
    prepared_notes_sha256="${prepared_notes_sha256%% *}"
    [ "$prepared_notes_sha256" = "$release_notes_sha256" ] || { print_error "prepared release-notes digest does not match its manifest"; exit 1; }
    [ "$(stat -f '%z' "$release_notes_copy")" = "$release_notes_size" ] || { print_error "prepared release-notes size does not match its manifest"; exit 1; }
    prepared_appcast_sha256="$(shasum -a 256 "$generated_appcast")"
    prepared_appcast_sha256="${prepared_appcast_sha256%% *}"
    [ "$prepared_appcast_sha256" = "$appcast_sha256" ] || { print_error "prepared appcast digest does not match its manifest"; exit 1; }
    [ "$(stat -f '%z' "$generated_appcast")" = "$appcast_size" ] || { print_error "prepared appcast size does not match its manifest"; exit 1; }
}

verify_github_release_asset() {
    matching_asset_count="$(printf '%s' "$release_assets_json" | jq --arg name "$release_asset_name" '[.assets[] | select(.name == $name)] | length')"
    [ "$matching_asset_count" = "1" ] || { print_error "GitHub Release must contain exactly one expected DMG"; exit 1; }
    [ "$(printf '%s' "$release_assets_json" | jq '.assets | length')" = "1" ] || { print_error "GitHub Release contains unexpected assets"; exit 1; }
    uploaded_asset_digest="$(printf '%s' "$release_assets_json" | jq --raw-output --arg name "$release_asset_name" '.assets[] | select(.name == $name) | .digest')"
    uploaded_asset_size="$(printf '%s' "$release_assets_json" | jq --raw-output --arg name "$release_asset_name" '.assets[] | select(.name == $name) | .size')"
    uploaded_asset_state="$(printf '%s' "$release_assets_json" | jq --raw-output --arg name "$release_asset_name" '.assets[] | select(.name == $name) | .state')"
    [ "$uploaded_asset_digest" = "sha256:${release_dmg_sha256}" ] || { print_error "GitHub Release asset digest does not match prepared DMG"; exit 1; }
    [ "$uploaded_asset_size" = "$release_dmg_size" ] || { print_error "GitHub Release asset size does not match prepared DMG"; exit 1; }
    [ "$uploaded_asset_state" = "uploaded" ] || { print_error "GitHub Release asset upload is incomplete"; exit 1; }
}

ensure_draft_release_asset() {
    github_asset_count="$(printf '%s' "$github_release_json" | jq '.assets | length')"
    github_expected_asset_count="$(printf '%s' "$github_release_json" | jq --arg name "$release_asset_name" \
        '[.assets[] | select(.name == $name)] | length')"
    if [ "$github_asset_count" = "0" ]; then
        gh release upload "$RELEASE_TAG" "$release_dmg" --repo "$repository_slug"
        load_github_release_state
        return
    fi
    [ "$github_asset_count" = "1" ] && [ "$github_expected_asset_count" = "1" ] || {
        print_error "draft GitHub Release contains unexpected assets"
        exit 1
    }
    uploaded_asset_digest="$(printf '%s' "$github_release_json" | jq --raw-output '.assets[0].digest')"
    uploaded_asset_size="$(printf '%s' "$github_release_json" | jq --raw-output '.assets[0].size')"
    uploaded_asset_state="$(printf '%s' "$github_release_json" | jq --raw-output '.assets[0].state')"
    if [ "$uploaded_asset_digest" != "sha256:${release_dmg_sha256}" ] \
        || [ "$uploaded_asset_size" != "$release_dmg_size" ] \
        || [ "$uploaded_asset_state" != "uploaded" ]; then
        gh release upload "$RELEASE_TAG" "$release_dmg" --repo "$repository_slug" --clobber
        load_github_release_state
    fi
}

verify_remote_release_tag() {
    remote_tag_reference="$(gh api "repos/${repository_slug}/git/ref/tags/${RELEASE_TAG}")"
    remote_object_type="$(printf '%s' "$remote_tag_reference" | jq --raw-output '.object.type')"
    remote_object_sha="$(printf '%s' "$remote_tag_reference" | jq --raw-output '.object.sha')"
    tag_depth=0
    while [ "$remote_object_type" = "tag" ]; do
        tag_depth=$((tag_depth + 1))
        [ "$tag_depth" -le 8 ] || { print_error "remote release tag nesting exceeds the safety limit"; exit 1; }
        remote_tag_object="$(gh api "repos/${repository_slug}/git/tags/${remote_object_sha}")"
        remote_object_type="$(printf '%s' "$remote_tag_object" | jq --raw-output '.object.type')"
        remote_object_sha="$(printf '%s' "$remote_tag_object" | jq --raw-output '.object.sha')"
    done
    [ "$remote_object_type" = "commit" ] && [ "$remote_object_sha" = "$current_commit" ] || {
        print_error "remote release tag does not resolve to the prepared commit"
        exit 1
    }
}

load_github_release_state() {
    github_release_json="$(gh release view "$RELEASE_TAG" --repo "$repository_slug" \
        --json assets,body,isDraft,isImmutable,isPrerelease,name,tagName)"
    [ "$(printf '%s' "$github_release_json" | jq --raw-output '.tagName')" = "$RELEASE_TAG" ] || {
        print_error "existing GitHub Release has a conflicting tag"
        exit 1
    }
    [ "$(printf '%s' "$github_release_json" | jq --raw-output '.name')" = "Astronomical ${release_version}" ] || {
        print_error "existing GitHub Release has a conflicting title"
        exit 1
    }
    expected_release_notes="$(cat "$release_notes_copy")"
    [ "$(printf '%s' "$github_release_json" | jq --raw-output '.body')" = "$expected_release_notes" ] || {
        print_error "existing GitHub Release has conflicting release notes"
        exit 1
    }
    github_release_is_draft="$(printf '%s' "$github_release_json" | jq --raw-output '.isDraft')"
    github_release_is_immutable="$(printf '%s' "$github_release_json" | jq --raw-output '.isImmutable')"
    [ "$(printf '%s' "$github_release_json" | jq --raw-output '.isPrerelease')" = "false" ] || {
        print_error "Stable publication cannot use a prerelease GitHub Release"
        exit 1
    }
    release_assets_json="$github_release_json"
}

publish_release() {
    repository_root="$1"
    start_step "verify-remote-release-tag"
    verify_remote_release_tag
    finish_step

    start_step "create-draft-github-release"
    if gh release view "$RELEASE_TAG" --repo "$repository_slug" >/dev/null 2>&1; then
        printf '%s\n' "Resuming matching GitHub Release ${RELEASE_TAG}."
    else
        gh release create "$RELEASE_TAG" "$release_dmg" \
            --repo "$repository_slug" \
            --verify-tag \
            --draft \
            --title "Astronomical ${release_version}" \
            --notes-file "$release_notes_copy"
    fi
    finish_step

    start_step "verify-and-publish-github-release"
    load_github_release_state
    if [ "$github_release_is_draft" = "true" ]; then
        [ "$github_release_is_immutable" = "false" ] || { print_error "draft GitHub Release is unexpectedly immutable"; exit 1; }
        ensure_draft_release_asset
        verify_github_release_asset
        gh release edit "$RELEASE_TAG" --repo "$repository_slug" --draft=false
        load_github_release_state
    fi
    [ "$github_release_is_draft" = "false" ] || { print_error "GitHub Release is still a draft"; exit 1; }
    verify_github_release_asset
    finish_step

    start_step "stage-signed-pages-appcast"
    install -m 0644 "$generated_appcast" "${repository_root}/site/appcast.xml"
    finish_step
}

main() {
    parse_arguments "$@"
    for command_name in cargo cat git grep install jq mktemp plutil shasum stat xmllint; do require_command "$command_name"; done
    if [ "$SHOULD_PUBLISH" = "true" ]; then require_command gh; fi
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"

    start_step "validate-release-identity"
    validate_release_identity "$repository_root"
    finish_step
    if [ "$SHOULD_PUBLISH" = "true" ]; then
        load_prepared_release
        publish_release "$repository_root"
    else
        require_command security
        require_command xcrun
        validate_distribution_credentials "$repository_root"
        prepare_release "$repository_root"
    fi

    printf '%s\n' "Prepared signed release: ${release_dmg}"
    if [ "$SHOULD_PUBLISH" = "true" ]; then
        printf '%s\n' "Published GitHub Release ${RELEASE_TAG}; review and commit site/appcast.xml to activate the update feed."
    else
        printf '%s\n' "No GitHub state changed. Review this exact directory, then re-run with the same output directory and --publish."
    fi
}

main "$@"
