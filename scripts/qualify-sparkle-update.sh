#!/usr/bin/env sh

# Proves signed installation, relaunch, external-state preservation, and corrupt
# archive rejection through Sparkle's own command-line update driver.

set -eu

readonly QUALIFICATION_TIMEOUT_SECONDS=120
readonly SPARKLE_PUBLIC_ED25519_KEY="g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ="
readonly QUALIFICATION_BUNDLE_IDENTIFIER="dev.astronomical.update-qualification"

QUALIFICATION_ROOT=""
SERVER_PROCESS_IDENTIFIER=""
CURRENT_APP_PROCESS_IDENTIFIER=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

start_step() {
    current_step_name="$1"
    current_step_started_at="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name"
}

finish_step() {
    printf '%s step=%s status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name" \
        "$(( $(date +%s) - current_step_started_at ))"
}

wait_for_file() {
    expected_file="$1"
    expected_description="$2"
    waited_seconds=0
    while [ ! -s "$expected_file" ]; do
        [ "$waited_seconds" -lt 30 ] || {
            print_error "timed out waiting for ${expected_description}"
            return 1
        }
        sleep 1
        waited_seconds=$((waited_seconds + 1))
        printf '%s step=%s status=waiting description="%s" elapsed_seconds=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name" \
            "$expected_description" "$waited_seconds"
    done
}

stop_qualified_app() {
    process_identifier_file="${QUALIFICATION_ROOT}/app.pid"
    [ -s "$process_identifier_file" ] || return 0
    CURRENT_APP_PROCESS_IDENTIFIER="$(tr -d '[:space:]' < "$process_identifier_file")"
    case "$CURRENT_APP_PROCESS_IDENTIFIER" in ''|*[!0-9]*) return 0 ;; esac
    process_command="$(ps -p "$CURRENT_APP_PROCESS_IDENTIFIER" -o command= 2>/dev/null || true)"
    case "$process_command" in
        *"${QUALIFICATION_ROOT}/installed/Astronomical Update Qualification.app/Contents/MacOS/qualification-app"*)
            kill -TERM "$CURRENT_APP_PROCESS_IDENTIFIER" 2>/dev/null || true
            ;;
    esac
    CURRENT_APP_PROCESS_IDENTIFIER=""
}

cleanup() {
    stop_qualified_app
    if [ -n "${SERVER_PROCESS_IDENTIFIER:-}" ]; then
        kill -TERM "$SERVER_PROCESS_IDENTIFIER" 2>/dev/null || true
        wait "$SERVER_PROCESS_IDENTIFIER" 2>/dev/null || true
    fi
    [ -n "${QUALIFICATION_ROOT:-}" ] && [ -d "$QUALIFICATION_ROOT" ] || return 0
    case "$QUALIFICATION_ROOT" in
        */target/sparkle-update-qualification.noindex) rm -rf "$QUALIFICATION_ROOT" ;;
        *) print_error "refusing to remove unexpected qualification directory: $QUALIFICATION_ROOT" ;;
    esac
}
trap cleanup 0

create_information_plist() {
    app_bundle="$1"
    bundle_version="$2"
    display_version="$3"
    feed_url="$4"
    launch_log="$5"
    process_identifier_file="$6"
    information_plist="${app_bundle}/Contents/Info.plist"

    plutil -create xml1 "$information_plist"
    plutil -insert CFBundleName -string "Astronomical Update Qualification" "$information_plist"
    plutil -insert CFBundleDisplayName -string "Astronomical Update Qualification" "$information_plist"
    plutil -insert CFBundleExecutable -string "qualification-app" "$information_plist"
    plutil -insert CFBundleIdentifier -string "$QUALIFICATION_BUNDLE_IDENTIFIER" "$information_plist"
    plutil -insert CFBundlePackageType -string "APPL" "$information_plist"
    plutil -insert CFBundleVersion -string "$bundle_version" "$information_plist"
    plutil -insert CFBundleShortVersionString -string "$display_version" "$information_plist"
    plutil -insert LSMinimumSystemVersion -string "26.0" "$information_plist"
    plutil -insert LSUIElement -bool true "$information_plist"
    plutil -insert SUFeedURL -string "$feed_url" "$information_plist"
    plutil -insert SUPublicEDKey -string "$SPARKLE_PUBLIC_ED25519_KEY" "$information_plist"
    plutil -insert SUVerifyUpdateBeforeExtraction -bool true "$information_plist"
    plutil -insert SURequireSignedFeed -bool true "$information_plist"
    plutil -insert SUSignedFeedFailureExpirationInterval -integer 0 "$information_plist"
    plutil -insert SUEnableAutomaticChecks -bool true "$information_plist"
    plutil -insert SUEnableSystemProfiling -bool false "$information_plist"
    plutil -insert AstronomicalQualificationLaunchLog -string "$launch_log" "$information_plist"
    plutil -insert AstronomicalQualificationPIDFile -string "$process_identifier_file" "$information_plist"
    plutil -insert NSAppTransportSecurity -xml '<dict><key>NSAllowsLocalNetworking</key><true/></dict>' "$information_plist"
}

assemble_fixture_app() {
    app_bundle="$1"
    bundle_version="$2"
    display_version="$3"
    feed_url="$4"
    qualification_executable="$5"
    sparkle_framework="$6"

    mkdir -p "${app_bundle}/Contents/MacOS" "${app_bundle}/Contents/Frameworks"
    create_information_plist \
        "$app_bundle" "$bundle_version" "$display_version" "$feed_url" \
        "${QUALIFICATION_ROOT}/launch.log" "${QUALIFICATION_ROOT}/app.pid"
    cp "$qualification_executable" "${app_bundle}/Contents/MacOS/qualification-app"
    chmod +x "${app_bundle}/Contents/MacOS/qualification-app"
    ditto "$sparkle_framework" "${app_bundle}/Contents/Frameworks/Sparkle.framework"
    codesign --force --sign - "${app_bundle}/Contents/MacOS/qualification-app"
    codesign --force --sign - --deep "${app_bundle}/Contents/Frameworks/Sparkle.framework"
    codesign --force --sign - "$app_bundle"
    codesign --verify --deep --strict "$app_bundle"
}

state_fingerprint() {
    state_directory="$1"
    shasum -a 256 \
        "${state_directory}/config/config.json" \
        "${state_directory}/models/model-marker" \
        "${state_directory}/cache/cache-marker" \
        "${state_directory}/logs/log-marker"
}

build_sparkle_cli() {
    sparkle_checkout="$1"
    sparkle_cli="${sparkle_checkout}/build/Release/sparkle.app/Contents/MacOS/sparkle"
    if [ ! -x "$sparkle_cli" ]; then
        xcodebuild \
            -project "${sparkle_checkout}/Sparkle.xcodeproj" \
            -target sparkle-cli \
            -configuration Release \
            ONLY_ACTIVE_ARCH=YES \
            ARCHS=arm64 \
            build >&2
    fi
    [ -x "$sparkle_cli" ] || { print_error "Sparkle command-line updater was not built"; exit 1; }
    printf '%s\n' "$sparkle_cli"
}

run_inner_qualification() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    QUALIFICATION_ROOT="${repository_root}/target/sparkle-update-qualification.noindex"
    [ ! -e "$QUALIFICATION_ROOT" ] || rm -rf "$QUALIFICATION_ROOT"
    mkdir -p "$QUALIFICATION_ROOT" "${QUALIFICATION_ROOT}/served" "${QUALIFICATION_ROOT}/state"

    for command_name in clang codesign curl ditto hdiutil plutil python3 shasum swift xcodebuild; do
        command -v "$command_name" >/dev/null 2>&1 || { print_error "required command is unavailable: $command_name"; exit 1; }
    done

    start_step "resolve-sparkle-tools"
    swift package resolve --package-path "${repository_root}/apps/astronomical-menu"
    swift build --configuration release --package-path "${repository_root}/apps/astronomical-menu"
    sparkle_checkout="${repository_root}/apps/astronomical-menu/.build/checkouts/Sparkle"
    sparkle_framework="${repository_root}/apps/astronomical-menu/.build/release/Sparkle.framework"
    appcast_tool="${repository_root}/apps/astronomical-menu/.build/artifacts/sparkle/Sparkle/bin/generate_appcast"
    sparkle_cli="$(build_sparkle_cli "$sparkle_checkout")"
    [ -d "$sparkle_framework" ] || { print_error "Sparkle framework is unavailable"; exit 1; }
    [ -x "$appcast_tool" ] || { print_error "Sparkle appcast tool is unavailable"; exit 1; }
    finish_step

    start_step "start-loopback-update-server"
    port_file="${QUALIFICATION_ROOT}/server.port"
    python3 "${repository_root}/scripts/fixtures/sparkle_update_loopback_server.py" \
        --directory "${QUALIFICATION_ROOT}/served" \
        --port-file "$port_file" &
    SERVER_PROCESS_IDENTIFIER="$!"
    wait_for_file "$port_file" "loopback server port"
    server_port="$(tr -d '[:space:]' < "$port_file")"
    feed_url="http://127.0.0.1:${server_port}/appcast.xml"
    finish_step

    start_step "assemble-old-and-new-apps"
    qualification_executable="${QUALIFICATION_ROOT}/qualification-app"
    clang -fobjc-arc -Wall -Werror -framework AppKit \
        "${repository_root}/scripts/fixtures/sparkle-update-qualification-app.m" \
        -o "$qualification_executable"
    old_app="${QUALIFICATION_ROOT}/old/Astronomical Update Qualification.app"
    new_app="${QUALIFICATION_ROOT}/new/Astronomical Update Qualification.app"
    assemble_fixture_app "$old_app" "1" "1.0.0" "$feed_url" "$qualification_executable" "$sparkle_framework"
    assemble_fixture_app "$new_app" "2" "2.0.0" "$feed_url" "$qualification_executable" "$sparkle_framework"
    hdiutil create \
        -volname "Astronomical Update Qualification" \
        -srcfolder "$new_app" \
        -format UDZO \
        "${QUALIFICATION_ROOT}/served/Astronomical-Update-Qualification-2.0.0.dmg"
    finish_step

    start_step "generate-signed-local-appcast"
    "$appcast_tool" \
        --download-url-prefix "http://127.0.0.1:${server_port}/" \
        --maximum-deltas 0 \
        -o "${QUALIFICATION_ROOT}/served/appcast.xml" \
        "${QUALIFICATION_ROOT}/served"
    grep -F 'sparkle:edSignature=' "${QUALIFICATION_ROOT}/served/appcast.xml" >/dev/null
    grep -F 'sparkle-signatures:' "${QUALIFICATION_ROOT}/served/appcast.xml" >/dev/null
    curl --silent --show-error --fail --max-time 5 "$feed_url" >/dev/null
    finish_step

    state_directory="${QUALIFICATION_ROOT}/state"
    mkdir -p "${state_directory}/config" "${state_directory}/models" "${state_directory}/cache" "${state_directory}/logs"
    printf '%s\n' '{"qualification":true}' > "${state_directory}/config/config.json"
    printf '%s\n' 'model artifact remains external' > "${state_directory}/models/model-marker"
    printf '%s\n' 'cache remains external' > "${state_directory}/cache/cache-marker"
    printf '%s\n' 'log remains external' > "${state_directory}/logs/log-marker"
    state_fingerprint "$state_directory" > "${QUALIFICATION_ROOT}/state.before.sha256"

    start_step "install-signed-update-and-relaunch"
    installed_app="${QUALIFICATION_ROOT}/installed/Astronomical Update Qualification.app"
    mkdir -p "${QUALIFICATION_ROOT}/installed"
    ditto "$old_app" "$installed_app"
    "${installed_app}/Contents/MacOS/qualification-app" &
    CURRENT_APP_PROCESS_IDENTIFIER="$!"
    wait_for_file "${QUALIFICATION_ROOT}/launch.log" "old application launch"
    "$sparkle_cli" "$installed_app" \
        --application "$installed_app" \
        --check-immediately \
        --feed-url "$feed_url" \
        --user-agent-name "Astronomical update qualification" \
        --verbose
    installed_version="$(plutil -extract CFBundleVersion raw -o - "${installed_app}/Contents/Info.plist")"
    [ "$installed_version" = "2" ] || { print_error "signed update did not install"; exit 1; }
    waited_seconds=0
    until grep -F '2' "${QUALIFICATION_ROOT}/launch.log" >/dev/null 2>&1; do
        [ "$waited_seconds" -lt 30 ] || { print_error "updated app did not relaunch"; exit 1; }
        sleep 1
        waited_seconds=$((waited_seconds + 1))
        printf '%s step=%s status=waiting description="updated app relaunch" elapsed_seconds=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_step_name" "$waited_seconds"
    done
    state_fingerprint "$state_directory" > "${QUALIFICATION_ROOT}/state.after-success.sha256"
    cmp "${QUALIFICATION_ROOT}/state.before.sha256" "${QUALIFICATION_ROOT}/state.after-success.sha256"
    stop_qualified_app
    finish_step

    start_step "reject-corrupted-update"
    rm -rf "$installed_app"
    ditto "$old_app" "$installed_app"
    printf '%s' 'corruption' >> "${QUALIFICATION_ROOT}/served/Astronomical-Update-Qualification-2.0.0.dmg"
    if "$sparkle_cli" "$installed_app" \
        --check-immediately \
        --feed-url "$feed_url" \
        --user-agent-name "Astronomical update qualification" \
        --verbose; then
        print_error "Sparkle accepted a corrupted update archive"
        exit 1
    fi
    installed_version="$(plutil -extract CFBundleVersion raw -o - "${installed_app}/Contents/Info.plist")"
    [ "$installed_version" = "1" ] || { print_error "corrupted update replaced the installed app"; exit 1; }
    state_fingerprint "$state_directory" > "${QUALIFICATION_ROOT}/state.after-corruption.sha256"
    cmp "${QUALIFICATION_ROOT}/state.before.sha256" "${QUALIFICATION_ROOT}/state.after-corruption.sha256"
    finish_step

    printf '%s\n' "Sparkle qualification passed: signed update installed and relaunched; corrupted update rejected; external state preserved."
}

main() {
    if [ "${ASTRONOMICAL_SPARKLE_QUALIFICATION_BOUNDED:-false}" != "true" ]; then
        printf '%s step=sparkle-update-qualification status=start timeout_seconds=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$QUALIFICATION_TIMEOUT_SECONDS"
        ASTRONOMICAL_SPARKLE_QUALIFICATION_BOUNDED=true \
            perl -e 'alarm shift; exec @ARGV' "$QUALIFICATION_TIMEOUT_SECONDS" "$0" "$@"
        return
    fi
    run_inner_qualification
}

main "$@"
