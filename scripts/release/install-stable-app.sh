#!/usr/bin/env sh

# Explicitly promotes one validated Stable bundle outside the repository build tree.

set -eu

SOURCE_APP_BUNDLE=""
DRY_RUN="false"
STAGED_APP_BUNDLE=""
BACKUP_APP_BUNDLE=""
CANONICAL_HOME_DIRECTORY=""
DESTINATION_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/release/install-stable-app.sh [--app-bundle PATH] [--dry-run]"
    printf '%s\n' "Installs a clean Stable bundle as ~/Applications/Astronomical.app."
}

cleanup() {
    # A staged candidate is disposable. A backup is the user's prior trusted
    # app and must survive interruption or a failed rollback for manual recovery.
    removable_bundle="${STAGED_APP_BUNDLE:-}"
    if [ -n "$removable_bundle" ] && [ -e "$removable_bundle" ]; then
        case "$removable_bundle" in
            "${DESTINATION_DIRECTORY:-}/.Astronomical.stage."*) rm -rf "$removable_bundle" ;;
            *) print_error "refusing to remove unexpected promotion path: ${removable_bundle}" ;;
        esac
    fi
}

validate_home_directory() {
    case "${HOME:-}" in
        /*) ;;
        *) print_error "HOME must be an absolute non-root directory"; exit 2 ;;
    esac
    case "/${HOME}/" in
        *"/../"*|*"/./"*) print_error "HOME must not contain parent or current-directory components"; exit 2 ;;
    esac
    [ -d "$HOME" ] || { print_error "HOME must identify an existing directory"; exit 2; }
    if ! CANONICAL_HOME_DIRECTORY="$(CDPATH='' cd -- "$HOME" && pwd -P)"; then
        print_error "HOME could not be resolved safely"
        exit 2
    fi
    [ "$CANONICAL_HOME_DIRECTORY" != "/" ] || {
        print_error "HOME must not resolve to the filesystem root"
        exit 2
    }
}
trap cleanup 0

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
    # Generated bundles live below a .noindex directory so Spotlight exposes
    # only the explicitly promoted copy in ~/Applications.
    SOURCE_APP_BUNDLE="${repository_root}/target/astronomical-macos-stable.noindex/Astronomical.app"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --app-bundle)
                [ "$#" -ge 2 ] || { print_error "--app-bundle requires a path"; exit 2; }
                SOURCE_APP_BUNDLE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
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
    validate_home_directory
    for required_command in plutil codesign ditto; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done
    [ -d "$SOURCE_APP_BUNDLE" ] || { print_error "Stable app bundle not found: ${SOURCE_APP_BUNDLE}"; exit 1; }
    for bundled_executable_name in astronomical-menu astronomicald astronomical-inference-worker; do
        bundled_executable="${SOURCE_APP_BUNDLE}/Contents/MacOS/${bundled_executable_name}"
        [ -x "$bundled_executable" ] || {
            print_error "required Stable executable is unavailable: ${bundled_executable_name}"
            exit 1
        }
    done

    validation_started_at="$(date +%s)"
    printf '%s step=validate-stable-bundle status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    application_channel="$(plutil -extract AstronomicalChannel raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    [ "$application_channel" = "stable" ] || { print_error "only a Stable bundle may be promoted"; exit 1; }
    application_version="$(plutil -extract CFBundleShortVersionString raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    application_build_number="$(plutil -extract CFBundleVersion raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    application_commit="$(plutil -extract AstronomicalBuildCommit raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    application_is_dirty="$(plutil -extract AstronomicalBuildDirty raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    application_build_date="$(plutil -extract AstronomicalBuildDate raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    bundle_identifier="$(plutil -extract CFBundleIdentifier raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    bundle_icon_file="$(plutil -extract CFBundleIconFile raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    supervisor_port="$(plutil -extract AstronomicalSupervisorPort raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    state_directory_name="$(plutil -extract AstronomicalStateDirectoryName raw -o - "${SOURCE_APP_BUNDLE}/Contents/Info.plist")"
    [ "$application_is_dirty" = "false" ] || { print_error "dirty builds cannot be promoted to Stable"; exit 1; }
    [ "$bundle_identifier" = "dev.astronomical.app" ] || { print_error "Stable bundle identifier is invalid"; exit 1; }
    [ "$supervisor_port" = "6732" ] || { print_error "Stable bundle must use port 6732"; exit 1; }
    [ "$state_directory_name" = ".astronomical" ] || { print_error "Stable bundle state directory is invalid"; exit 1; }
    case "$application_build_number" in ''|*[!0-9]*) print_error "Stable bundle build number must be numeric"; exit 1 ;; esac
    case "$application_build_date" in ????????) ;; *) print_error "Stable bundle build date must use YYYYMMDD"; exit 1 ;; esac
    case "$application_build_date" in *[!0-9]*) print_error "Stable bundle build date must use YYYYMMDD"; exit 1 ;; esac
    [ -n "$application_version" ] || { print_error "Stable bundle version is unavailable"; exit 1; }
    [ -n "$application_commit" ] || { print_error "Stable bundle commit is unavailable"; exit 1; }
    [ "$bundle_icon_file" = "Astronomical.icns" ] || { print_error "Stable bundle icon identity is invalid"; exit 1; }
    for packaged_resource in LICENSE THIRD_PARTY_NOTICES RUST_DEPENDENCY_NOTICES Astronomical.icns; do
        [ -s "${SOURCE_APP_BUNDLE}/Contents/Resources/${packaged_resource}" ] || {
            print_error "required bundled resource is unavailable: ${packaged_resource}"
            exit 1
        }
    done
    codesign --verify --deep --strict "$SOURCE_APP_BUNDLE"
    daemon_version_output="$("${SOURCE_APP_BUNDLE}/Contents/MacOS/astronomicald" --version)"
    case "$daemon_version_output" in
        *"${application_version}"*"${application_commit}"*) ;;
        *) print_error "Stable app and bundled daemon identities do not match"; exit 1 ;;
    esac
    printf '%s step=validate-stable-bundle status=success elapsed_seconds=%s version=%s commit=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(( $(date +%s) - validation_started_at ))" \
        "$application_version" "$application_commit"

    DESTINATION_DIRECTORY="${CANONICAL_HOME_DIRECTORY}/Applications"
    destination_app_bundle="${DESTINATION_DIRECTORY}/Astronomical.app"
    if [ "$DRY_RUN" = "true" ]; then
        printf '%s\n' "Would install ${application_version} Stable (${application_commit}) at ${destination_app_bundle}"
        return
    fi

    promotion_started_at="$(date +%s)"
    printf '%s step=stage-and-promote-stable status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    mkdir -p "$DESTINATION_DIRECTORY"
    STAGED_APP_BUNDLE="${DESTINATION_DIRECTORY}/.Astronomical.stage.$$"
    BACKUP_APP_BUNDLE="${DESTINATION_DIRECTORY}/.Astronomical.backup.$$"
    [ ! -e "$STAGED_APP_BUNDLE" ] || { print_error "promotion staging path already exists"; exit 1; }
    [ ! -e "$BACKUP_APP_BUNDLE" ] || { print_error "promotion backup path already exists"; exit 1; }
    ditto "$SOURCE_APP_BUNDLE" "$STAGED_APP_BUNDLE"
    codesign --verify --deep --strict "$STAGED_APP_BUNDLE"
    if [ -e "$destination_app_bundle" ]; then
        mv "$destination_app_bundle" "$BACKUP_APP_BUNDLE"
    fi
    if ! mv "$STAGED_APP_BUNDLE" "$destination_app_bundle"; then
        if [ -e "$BACKUP_APP_BUNDLE" ]; then
            if mv "$BACKUP_APP_BUNDLE" "$destination_app_bundle"; then
                BACKUP_APP_BUNDLE=""
                print_error "Stable promotion failed; the prior app was restored"
            else
                print_error "Stable promotion and rollback failed; the prior app is preserved at ${BACKUP_APP_BUNDLE}"
            fi
        else
            print_error "Stable promotion failed before an existing app could be backed up"
        fi
        exit 1
    fi
    STAGED_APP_BUNDLE=""
    if [ -e "$BACKUP_APP_BUNDLE" ]; then
        rm -rf "$BACKUP_APP_BUNDLE"
        BACKUP_APP_BUNDLE=""
    fi
    printf '%s step=stage-and-promote-stable status=success elapsed_seconds=%s destination=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(( $(date +%s) - promotion_started_at ))" \
        "$destination_app_bundle"
    printf '%s\n' "Installed ${application_version} Stable (${application_commit}). The running app was not restarted."
}

main "$@"
