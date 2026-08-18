#!/usr/bin/env sh

# Exercises Stable promotion and rollback with fake signing and metadata tools.

set -eu

readonly INSTALLER_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe installer test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

prepare_existing_stable_app() {
    applications_directory="$1"
    rm -rf "$applications_directory"
    mkdir -p "${applications_directory}/Astronomical.app"
    printf '%s\n' "trusted-prior-build" > "${applications_directory}/Astronomical.app/prior-build-marker"
}

run_installer() {
    failure_mode="$1"
    output_file="$2"
    : > "$movement_count_file"
    HOME="$test_home_directory" PATH="${fake_command_directory}:${PATH}" \
        MOVEMENT_COUNT_FILE="$movement_count_file" FAKE_MV_FAILURE_MODE="$failure_mode" \
        FAKE_APPLICATION_VERSION="${FAKE_APPLICATION_VERSION-0.2.0}" \
        FAKE_APPLICATION_COMMIT="${FAKE_APPLICATION_COMMIT-abcdef123456}" \
        timeout "$INSTALLER_TIMEOUT_SECONDS" "$installer_script" --app-bundle "$source_app_bundle" \
        > "$output_file" 2>&1
}

run_installer_with_default_source() {
    output_file="$1"
    : > "$movement_count_file"
    HOME="$test_home_directory" PATH="${fake_command_directory}:${PATH}" \
        MOVEMENT_COUNT_FILE="$movement_count_file" FAKE_MV_FAILURE_MODE="none" \
        FAKE_APPLICATION_VERSION="0.2.0" FAKE_APPLICATION_COMMIT="abcdef123456" \
        timeout "$INSTALLER_TIMEOUT_SECONDS" "$installer_script" > "$output_file" 2>&1
}

assert_no_promotion_artifacts() {
    applications_directory="$1"
    for promotion_artifact in \
        "${applications_directory}"/.Astronomical.stage.* \
        "${applications_directory}"/.Astronomical.backup.*
    do
        [ ! -e "$promotion_artifact" ] || {
            print_error "unexpected promotion artifact remained: ${promotion_artifact}"
            exit 1
        }
    done
}

main() {
    for required_command in timeout mktemp grep; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-stable-installer.XXXXXX")"
    sandbox_repository="${SANDBOX_DIRECTORY}/repository"
    mkdir -p "${sandbox_repository}/scripts/release"
    cp "${repository_root}/scripts/release/install-stable-app.sh" \
        "${sandbox_repository}/scripts/release/install-stable-app.sh"
    chmod +x "${sandbox_repository}/scripts/release/install-stable-app.sh"
    installer_script="${sandbox_repository}/scripts/release/install-stable-app.sh"
    test_home_directory="${SANDBOX_DIRECTORY}/home"
    applications_directory="${test_home_directory}/Applications"
    source_app_bundle="${SANDBOX_DIRECTORY}/candidate/Astronomical.app"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    movement_count_file="${SANDBOX_DIRECTORY}/movement-count"
    mkdir -p "${source_app_bundle}/Contents/MacOS" "${source_app_bundle}/Contents/Resources" \
        "$fake_command_directory"
    printf '%s\n' "plist-fixture" > "${source_app_bundle}/Contents/Info.plist"
    printf '%s\n' "new-stable-build" > "${source_app_bundle}/new-build-marker"
    for packaged_resource in LICENSE THIRD_PARTY_NOTICES RUST_DEPENDENCY_NOTICES Astronomical.icns; do
        printf '%s\n' "fixture" > "${source_app_bundle}/Contents/Resources/${packaged_resource}"
    done
    cat > "${source_app_bundle}/Contents/MacOS/astronomicald" <<'DAEMON'
#!/usr/bin/env sh
printf '%s\n' 'astronomicald 0.2.0 (build 107, abcdef123456)'
DAEMON
    cat > "${source_app_bundle}/Contents/MacOS/astronomical-menu" <<'MENU'
#!/usr/bin/env sh
exit 0
MENU
    cp "${source_app_bundle}/Contents/MacOS/astronomical-menu" \
        "${source_app_bundle}/Contents/MacOS/astronomical-inference-worker"
    cat > "${fake_command_directory}/plutil" <<'PLUTIL'
#!/usr/bin/env sh
case "$2" in
    AstronomicalChannel) printf '%s\n' stable ;;
    CFBundleShortVersionString) printf '%s\n' "${FAKE_APPLICATION_VERSION-0.2.0}" ;;
    CFBundleVersion) printf '%s\n' 107 ;;
    AstronomicalBuildCommit) printf '%s\n' "${FAKE_APPLICATION_COMMIT-abcdef123456}" ;;
    AstronomicalBuildDirty) printf '%s\n' false ;;
    AstronomicalBuildDate) printf '%s\n' 20260814 ;;
    CFBundleIdentifier) printf '%s\n' dev.astronomical.app ;;
    CFBundleIconFile) printf '%s\n' Astronomical.icns ;;
    AstronomicalSupervisorPort) printf '%s\n' 6732 ;;
    AstronomicalStateDirectoryName) printf '%s\n' .astronomical ;;
    *) exit 1 ;;
esac
PLUTIL
    cat > "${fake_command_directory}/codesign" <<'CODESIGN'
#!/usr/bin/env sh
exit 0
CODESIGN
    cat > "${fake_command_directory}/ditto" <<'DITTO'
#!/usr/bin/env sh
exec /bin/cp -R "$1" "$2"
DITTO
    cat > "${fake_command_directory}/mv" <<'MOVE'
#!/usr/bin/env sh
movement_count=0
if [ -s "${MOVEMENT_COUNT_FILE:?MOVEMENT_COUNT_FILE is required}" ]; then
    IFS= read -r movement_count < "$MOVEMENT_COUNT_FILE"
fi
movement_count=$((movement_count + 1))
printf '%s\n' "$movement_count" > "$MOVEMENT_COUNT_FILE"
if [ "${FAKE_MV_FAILURE_MODE:-}" = "stage" ] && [ "$movement_count" -eq 2 ]; then
    exit 23
fi
if [ "${FAKE_MV_FAILURE_MODE:-}" = "stage-and-rollback" ] \
    && { [ "$movement_count" -eq 2 ] || [ "$movement_count" -eq 3 ]; }
then
    exit 23
fi
exec /bin/mv "$@"
MOVE
    chmod +x "${source_app_bundle}/Contents/MacOS/astronomicald" "$fake_command_directory"/*
    chmod +x "${source_app_bundle}/Contents/MacOS/astronomical-menu" \
        "${source_app_bundle}/Contents/MacOS/astronomical-inference-worker"
    mkdir -p "$test_home_directory"

    printf '%s\n' '[stable-installer-test] case=noncanonical-home-is-rejected status=start'
    if HOME="${test_home_directory}/.." PATH="${fake_command_directory}:${PATH}" \
        timeout "$INSTALLER_TIMEOUT_SECONDS" "$installer_script" \
        --app-bundle "$source_app_bundle" --dry-run \
        > "${SANDBOX_DIRECTORY}/noncanonical-home.log" 2>&1
    then
        print_error "installer unexpectedly accepted a HOME path containing a parent component"
        exit 1
    else
        home_validation_exit_code=$?
    fi
    [ "$home_validation_exit_code" -eq 2 ] || {
        print_error "HOME validation returned ${home_validation_exit_code}, expected 2"
        exit 1
    }
    printf '%s\n' '[stable-installer-test] case=noncanonical-home-is-rejected status=success'

    printf '%s\n' '[stable-installer-test] case=missing-worker-is-rejected status=start'
    rm "${source_app_bundle}/Contents/MacOS/astronomical-inference-worker"
    if run_installer "none" "${SANDBOX_DIRECTORY}/missing-worker.log"; then
        print_error "installer unexpectedly accepted a bundle without its worker"
        exit 1
    fi
    grep -F "required Stable executable is unavailable: astronomical-inference-worker" \
        "${SANDBOX_DIRECTORY}/missing-worker.log" >/dev/null || {
            print_error "missing worker rejection did not identify the worker"
            exit 1
        }
    cp "${source_app_bundle}/Contents/MacOS/astronomical-menu" \
        "${source_app_bundle}/Contents/MacOS/astronomical-inference-worker"
    printf '%s\n' '[stable-installer-test] case=missing-worker-is-rejected status=success'

    printf '%s\n' '[stable-installer-test] case=missing-version-is-rejected status=start'
    FAKE_APPLICATION_VERSION=""
    if run_installer "none" "${SANDBOX_DIRECTORY}/missing-version.log"; then
        print_error "installer unexpectedly accepted an empty Stable version"
        exit 1
    fi
    grep -F "Stable bundle version is unavailable" \
        "${SANDBOX_DIRECTORY}/missing-version.log" >/dev/null || {
            print_error "missing version rejection was unclear"
            exit 1
        }
    unset FAKE_APPLICATION_VERSION
    printf '%s\n' '[stable-installer-test] case=missing-version-is-rejected status=success'

    printf '%s\n' '[stable-installer-test] case=missing-commit-is-rejected status=start'
    FAKE_APPLICATION_COMMIT=""
    if run_installer "none" "${SANDBOX_DIRECTORY}/missing-commit.log"; then
        print_error "installer unexpectedly accepted an empty Stable commit"
        exit 1
    fi
    grep -F "Stable bundle commit is unavailable" \
        "${SANDBOX_DIRECTORY}/missing-commit.log" >/dev/null || {
            print_error "missing commit rejection was unclear"
            exit 1
        }
    unset FAKE_APPLICATION_COMMIT
    printf '%s\n' '[stable-installer-test] case=missing-commit-is-rejected status=success'

    printf '%s\n' '[stable-installer-test] case=successful-promotion status=start'
    prepare_existing_stable_app "$applications_directory"
    run_installer "none" "${SANDBOX_DIRECTORY}/successful-promotion.log"
    [ -f "${applications_directory}/Astronomical.app/new-build-marker" ] || {
        print_error "new Stable candidate was not installed"
        exit 1
    }
    [ -s "${applications_directory}/Astronomical.app/Contents/Resources/Astronomical.icns" ] || {
        print_error "new Stable candidate did not include its macOS icon"
        exit 1
    }
    assert_no_promotion_artifacts "$applications_directory"
    printf '%s\n' '[stable-installer-test] case=successful-promotion status=success'

    printf '%s\n' '[stable-installer-test] case=default-noindex-source-is-promoted status=start'
    default_source_app_bundle="${sandbox_repository}/target/astronomical-macos-stable.noindex/Astronomical.app"
    mkdir -p "$(dirname "$default_source_app_bundle")"
    cp -R "$source_app_bundle" "$default_source_app_bundle"
    prepare_existing_stable_app "$applications_directory"
    run_installer_with_default_source "${SANDBOX_DIRECTORY}/default-source-promotion.log"
    [ -f "${applications_directory}/Astronomical.app/new-build-marker" ] || {
        print_error "the default .noindex Stable candidate was not installed"
        exit 1
    }
    [ -d "$default_source_app_bundle" ] || {
        print_error "the installer unexpectedly removed its source build artifact"
        exit 1
    }
    assert_no_promotion_artifacts "$applications_directory"
    printf '%s\n' '[stable-installer-test] case=default-noindex-source-is-promoted status=success'

    printf '%s\n' '[stable-installer-test] case=successful-rollback status=start'
    prepare_existing_stable_app "$applications_directory"
    if run_installer "stage" "${SANDBOX_DIRECTORY}/successful-rollback.log"; then
        print_error "promotion unexpectedly succeeded after staged move failure"
        exit 1
    fi
    [ -f "${applications_directory}/Astronomical.app/prior-build-marker" ] || {
        print_error "prior Stable app was not restored"
        exit 1
    }
    assert_no_promotion_artifacts "$applications_directory"
    printf '%s\n' '[stable-installer-test] case=successful-rollback status=success'

    printf '%s\n' '[stable-installer-test] case=failed-rollback-preserves-backup status=start'
    prepare_existing_stable_app "$applications_directory"
    if run_installer "stage-and-rollback" "${SANDBOX_DIRECTORY}/failed-rollback.log"; then
        print_error "promotion unexpectedly succeeded after rollback failure"
        exit 1
    fi
    preserved_backup_count=0
    for preserved_backup in "${applications_directory}"/.Astronomical.backup.*; do
        [ -e "$preserved_backup" ] || continue
        preserved_backup_count=$((preserved_backup_count + 1))
        [ -f "${preserved_backup}/prior-build-marker" ] || {
            print_error "preserved backup does not contain the prior Stable app"
            exit 1
        }
    done
    [ "$preserved_backup_count" -eq 1 ] || {
        print_error "expected one preserved prior Stable backup, found ${preserved_backup_count}"
        exit 1
    }
    for staged_candidate in "${applications_directory}"/.Astronomical.stage.*; do
        [ ! -e "$staged_candidate" ] || {
            print_error "failed promotion left a disposable staged candidate behind"
            exit 1
        }
    done
    if ! grep -F "the prior app is preserved at" "${SANDBOX_DIRECTORY}/failed-rollback.log" >/dev/null; then
        print_error "rollback failure did not report the preserved backup location"
        exit 1
    fi
    printf '%s\n' '[stable-installer-test] case=failed-rollback-preserves-backup status=success'
}

main "$@"
