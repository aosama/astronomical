#!/usr/bin/env sh

# Builds one channel-specific optimized app bundle, then validates only that instance.
#
# Exit codes:
#   0 — build and validation succeeded
#   1 — build or validation failure
#
# Every phase reports its number, start, end, elapsed time, and ETA.

set -eu

# ── Constants ──────────────────────────────────────────────────────────

readonly TOTAL_PHASES=5
readonly SPARKLE_PUBLIC_ED25519_KEY="g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ="
APPLICATION_CHANNEL="development"
SIGNING_IDENTITY=""

# ── Helpers ────────────────────────────────────────────────────────────

script_start_seconds=0
current_phase=0

print_usage() {
    printf '%s\n' "Usage: scripts/internal/build-macos-app.sh --channel development|stable [--signing-identity NAME]"
    printf '%s\n' ""
    printf '%s\n' "Builds release binaries, assembles Astronomical.app, and runs"
    printf '%s\n' "post-build validation. Generated apps use .noindex output directories"
    printf '%s\n' "so Spotlight exposes only explicitly installed applications."
    printf '%s\n' "Development is the safe default. Stable builds require a clean worktree."
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

require_command() {
    required_command="$1"
    if ! command -v "$required_command" >/dev/null 2>&1; then
        print_error "required command is unavailable: $required_command"
        exit 1
    fi
}

start_phase() {
    current_phase="$1"
    phase_description="$2"
    script_start_seconds="${script_start_seconds:-0}"
    if [ "$script_start_seconds" -eq 0 ]; then
        script_start_seconds="$(date +%s)"
    fi
    phase_start_seconds="$(date +%s)"
    printf '%s phase=%s/%s status=start description="%s"\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_phase" "$TOTAL_PHASES" "$phase_description"
}

finish_phase() {
    phase_status="$1"
    phase_finish_seconds="$(date +%s)"
    phase_elapsed_seconds=$((phase_finish_seconds - phase_start_seconds))
    script_elapsed_seconds=$((phase_finish_seconds - script_start_seconds))
    printf '%s phase=%s/%s status=%s elapsed_seconds=%s total_elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$current_phase" "$TOTAL_PHASES" \
        "$phase_status" "$phase_elapsed_seconds" "$script_elapsed_seconds"
}

remove_previous_app_bundle() {
    bundle_path="$1"
    case "$bundle_path" in
        *"/target/astronomical-macos-development.noindex/Astronomical Development.app"|*"/target/astronomical-macos-stable.noindex/Astronomical.app")
            rm -rf "$bundle_path"
            ;;
        *)
            print_error "refusing to remove unexpected app bundle path: $bundle_path"
            exit 1
            ;;
    esac
}

github_pages_feed_url() {
    repository_url="$1"
    case "$repository_url" in
        https://github.com/*/*) ;;
        *)
            print_error "workspace repository must be an HTTPS GitHub URL to derive the update feed"
            exit 1
            ;;
    esac
    repository_slug="${repository_url#https://github.com/}"
    repository_slug="${repository_slug%.git}"
    repository_owner="${repository_slug%%/*}"
    repository_name="${repository_slug#*/}"
    [ -n "$repository_owner" ] && [ -n "$repository_name" ] && [ "$repository_name" != "$repository_slug" ] || {
        print_error "workspace repository URL does not contain an owner and repository"
        exit 1
    }
    printf 'https://%s.github.io/%s/appcast.xml\n' "$repository_owner" "$repository_name"
}

sign_code_object() {
    code_object_path="$1"
    [ -e "$code_object_path" ] || {
        print_error "required code object is unavailable: ${code_object_path}"
        exit 1
    }
    code_object_name="$(basename -- "$code_object_path")"
    signing_started_at="$(date +%s)"
    printf '%s operation=code-sign object="%s" status=start\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$code_object_name"
    if [ -n "$SIGNING_IDENTITY" ]; then
        codesign --force --sign "$SIGNING_IDENTITY" --options runtime --timestamp \
            --preserve-metadata=entitlements "$code_object_path"
    else
        codesign --force --sign - --preserve-metadata=entitlements "$code_object_path"
    fi
    printf '%s operation=code-sign object="%s" status=success elapsed_seconds=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$code_object_name" \
        "$(( $(date +%s) - signing_started_at ))"
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
    if [ "${ASTRONOMICAL_CARGO_TARGET_LIFECYCLE:-}" != "disposable" ]; then
        exec "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
            --lane app-release -- \
            "${repository_root}/scripts/internal/build-macos-app.sh" "$@"
    fi
    [ -n "${CARGO_TARGET_DIR:-}" ] || {
        print_error "disposable app build is missing CARGO_TARGET_DIR"
        exit 1
    }

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --channel)
                if [ "$#" -lt 2 ]; then
                    print_error "--channel requires development or stable"
                    exit 2
                fi
                APPLICATION_CHANNEL="$2"
                shift 2
                continue
                ;;
            --signing-identity)
                [ "$#" -ge 2 ] || { print_error "--signing-identity requires a value"; exit 2; }
                SIGNING_IDENTITY="$2"
                shift 2
                continue
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "unrecognized argument: $1"
                print_usage >&2
                exit 1
                ;;
        esac
    done

    script_start_seconds="$(date +%s)"

    printf '%s\n' ""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' "  Astronomical ${APPLICATION_CHANNEL} Build Pipeline (${TOTAL_PHASES} phases)"
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' ""

    # ── Phase 1: Validate prerequisites ──────────────────────────────────

    start_phase 1 "validate prerequisites"
    require_command cmake
    require_command xcrun
    require_command cargo
    require_command rustc
    require_command swift
    require_command sysctl
    require_command codesign
    require_command ditto
    require_command plutil
    require_command iconutil
    require_command install_name_tool
    require_command git
    require_command jq
    finish_phase "success"

    case "$APPLICATION_CHANNEL" in
        development)
            # The .noindex suffix keeps this generated app out of Spotlight
            # while retaining it for direct launch and validation.
            release_directory="${repository_root}/target/astronomical-macos-development.noindex"
            app_bundle_path="${release_directory}/Astronomical Development.app"
            bundle_name="Astronomical Development"
            bundle_identifier="dev.astronomical.app.development"
            supervisor_port="6733"
            state_directory_name=".astronomical-dev"
            ;;
        stable)
            # Only the promoted ~/Applications copy should appear in Spotlight.
            release_directory="${repository_root}/target/astronomical-macos-stable.noindex"
            app_bundle_path="${release_directory}/Astronomical.app"
            bundle_name="Astronomical"
            bundle_identifier="dev.astronomical.app"
            supervisor_port="6732"
            state_directory_name=".astronomical"
            ;;
        *)
            print_error "channel must be development or stable"
            exit 2
            ;;
    esac
    if [ -n "$SIGNING_IDENTITY" ] && [ "$APPLICATION_CHANNEL" != "stable" ]; then
        print_error "Developer ID signing is reserved for Stable release bundles"
        exit 2
    fi
    cargo_workspace_metadata="$(cargo metadata --no-deps --format-version 1)"
    application_version="$(printf '%s' "$cargo_workspace_metadata" | jq --raw-output '.packages[] | select(.name == "astronomical-supervisor") | .version')"
    application_repository_url="$(printf '%s' "$cargo_workspace_metadata" | jq --raw-output '.packages[] | select(.name == "astronomical-supervisor") | .repository')"
    sparkle_update_feed_url="${ASTRONOMICAL_UPDATE_FEED_URL:-$(github_pages_feed_url "$application_repository_url")}"
    build_commit="$(git rev-parse --short=12 HEAD)"
    build_number="$(git rev-list --count HEAD)"
    # The icon and bundle metadata use one UTC date so their visible and
    # machine-readable build identities cannot disagree around local midnight.
    build_date_utc="$(date -u '+%Y%m%d')"
    if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
        build_dirty="true"
    else
        build_dirty="false"
    fi
    if [ "$APPLICATION_CHANNEL" = "stable" ] && [ "$build_dirty" = "true" ]; then
        print_error "Stable builds require a clean worktree; build Development or commit the intended changes"
        exit 1
    fi
    export ASTRONOMICAL_BUILD_COMMIT="$build_commit"
    export ASTRONOMICAL_BUILD_NUMBER="$build_number"
    export ASTRONOMICAL_BUILD_DIRTY="$build_dirty"

    # ── Phase 2: Verify native dependencies ───────────────────────────────

    start_phase 2 "verify native MLX dependencies"
    "${repository_root}/scripts/bootstrap-native-dependencies.sh" --verify
    finish_phase "success"

    # ── Phase 3: Build release binaries ──────────────────────────────────

    start_phase 3 "build optimized release binaries and native menu application"
    logical_cpu_count="$(sysctl -n hw.logicalcpu)"
    host_target_triple="$(rustc --print host-tuple)"
    case "$host_target_triple" in
        ''|*[!A-Za-z0-9_.-]*)
            print_error "rustc did not return a valid host target triple"
            exit 1
            ;;
    esac
    export CARGO_BUILD_JOBS="$logical_cpu_count"
    if command -v sccache >/dev/null 2>&1; then
        export RUSTC_WRAPPER=sccache
        printf '  using sccache wrapper, %s parallel jobs\n' "$logical_cpu_count"
    else
        printf '  %s parallel jobs (no sccache detected)\n' "$logical_cpu_count"
    fi
    # Cargo prints its own live compilation progress to stderr.
    cargo build --release --target "$host_target_triple" \
        -p astronomical-inference-worker --bin astronomical-inference-worker \
        -p astronomical-supervisor --bin astronomicald
    swift build --configuration release --package-path "${repository_root}/apps/astronomical-menu"
    finish_phase "success"

    # ── Phase 4: Assemble and sign app bundle ────────────────────────────

    start_phase 4 "assemble and codesign app bundle"
    remove_previous_app_bundle "$app_bundle_path"
    mkdir -p "${app_bundle_path}/Contents/MacOS"
    mkdir -p "${app_bundle_path}/Contents/Frameworks"
    mkdir -p "${app_bundle_path}/Contents/Resources"

    {
        printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
        printf '%s\n' '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
        printf '%s\n' '<plist version="1.0">'
        printf '%s\n' '<dict>'
        printf '  <key>CFBundleName</key><string>%s</string>\n' "$bundle_name"
        printf '  <key>CFBundleDisplayName</key><string>%s</string>\n' "$bundle_name"
        printf '%s\n' '  <key>CFBundleExecutable</key><string>astronomical-menu</string>'
        printf '  <key>CFBundleIdentifier</key><string>%s</string>\n' "$bundle_identifier"
        printf '  <key>CFBundleVersion</key><string>%s</string>\n' "$build_number"
        printf '  <key>CFBundleShortVersionString</key><string>%s</string>\n' "$application_version"
        printf '  <key>LSMinimumSystemVersion</key><string>26.0</string>\n'
        printf '%s\n' '  <key>CFBundleIconFile</key><string>Astronomical.icns</string>'
        printf '  <key>SUFeedURL</key><string>%s</string>\n' "$sparkle_update_feed_url"
        printf '  <key>SUPublicEDKey</key><string>%s</string>\n' "$SPARKLE_PUBLIC_ED25519_KEY"
        printf '%s\n' '  <key>SUVerifyUpdateBeforeExtraction</key><true/>'
        printf '%s\n' '  <key>SURequireSignedFeed</key><true/>'
        printf '%s\n' '  <key>SUSignedFeedFailureExpirationInterval</key><integer>0</integer>'
        printf '%s\n' '  <key>SUEnableAutomaticChecks</key><true/>'
        printf '%s\n' '  <key>SUScheduledCheckInterval</key><integer>86400</integer>'
        printf '%s\n' '  <key>SUAutomaticallyUpdate</key><false/>'
        printf '%s\n' '  <key>SUEnableSystemProfiling</key><false/>'
        printf '  <key>AstronomicalChannel</key><string>%s</string>\n' "$APPLICATION_CHANNEL"
        printf '  <key>AstronomicalBuildDate</key><string>%s</string>\n' "$build_date_utc"
        printf '  <key>AstronomicalSupervisorPort</key><integer>%s</integer>\n' "$supervisor_port"
        printf '  <key>AstronomicalStateDirectoryName</key><string>%s</string>\n' "$state_directory_name"
        printf '  <key>AstronomicalBuildCommit</key><string>%s</string>\n' "$build_commit"
        if [ "$build_dirty" = "true" ]; then
            printf '%s\n' '  <key>AstronomicalBuildDirty</key><true/>'
        else
            printf '%s\n' '  <key>AstronomicalBuildDirty</key><false/>'
        fi
        printf '%s\n' '  <key>CFBundlePackageType</key><string>APPL</string>'
        printf '%s\n' '  <key>LSUIElement</key><true/>'
        printf '%s\n' '</dict>'
        printf '%s\n' '</plist>'
    } > "${app_bundle_path}/Contents/Info.plist"

    cp "${repository_root}/apps/astronomical-menu/.build/release/astronomical-menu" \
       "${app_bundle_path}/Contents/MacOS/astronomical-menu"
    cargo_release_directory="${CARGO_TARGET_DIR}/${host_target_triple}/release"
    cp "${cargo_release_directory}/astronomicald" \
       "${app_bundle_path}/Contents/MacOS/astronomicald"
    cp "${cargo_release_directory}/astronomical-inference-worker" \
       "${app_bundle_path}/Contents/MacOS/astronomical-inference-worker"
    cp "${repository_root}/LICENSE" \
       "${app_bundle_path}/Contents/Resources/LICENSE"
    cp "${repository_root}/third-party/THIRD_PARTY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/THIRD_PARTY_NOTICES"
    cp "${repository_root}/third-party/RUST_DEPENDENCY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/RUST_DEPENDENCY_NOTICES"
    sparkle_license_source="${repository_root}/apps/astronomical-menu/.build/checkouts/Sparkle/LICENSE"
    [ -s "$sparkle_license_source" ] || {
        print_error "Sparkle license is unavailable after the Swift package build: ${sparkle_license_source}"
        exit 1
    }
    cp "$sparkle_license_source" "${app_bundle_path}/Contents/Resources/SPARKLE_LICENSE"
    sparkle_framework_source="${repository_root}/apps/astronomical-menu/.build/release/Sparkle.framework"
    sparkle_framework_destination="${app_bundle_path}/Contents/Frameworks/Sparkle.framework"
    [ -d "$sparkle_framework_source" ] || {
        print_error "Sparkle framework is unavailable after the Swift package build: ${sparkle_framework_source}"
        exit 1
    }
    ditto "$sparkle_framework_source" "$sparkle_framework_destination"

    # Stable keeps one timeless product identity; Development remains visibly distinct.
    iconset_directory="${app_bundle_path}/Contents/Resources/Astronomical.iconset"
    icon_resource="${app_bundle_path}/Contents/Resources/Astronomical.icns"
    swift "${repository_root}/scripts/internal/render-macos-app-icon.swift" \
        --output-directory "$iconset_directory" \
        --channel "$APPLICATION_CHANNEL"
    iconutil --convert icns --output "$icon_resource" "$iconset_directory"
    case "$iconset_directory" in
        "${app_bundle_path}/Contents/Resources/Astronomical.iconset") rm -rf "$iconset_directory" ;;
        *) print_error "refusing to remove unexpected iconset path: ${iconset_directory}"; exit 1 ;;
    esac
    [ -s "$icon_resource" ] || {
        print_error "generated macOS icon is unavailable: ${icon_resource}"
        exit 1
    }
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomical-menu"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomicald"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomical-inference-worker"
    # Swift Package Manager links Sparkle through @loader_path. The packaged
    # application keeps frameworks in Contents/Frameworks, so add the standard
    # bundle rpath before signing the executable and enclosing bundle.
    install_name_tool -add_rpath "@executable_path/../Frameworks" \
        "${app_bundle_path}/Contents/MacOS/astronomical-menu"

    printf '  signing embedded Sparkle code inside-out...\n'
    sparkle_version_directory="${sparkle_framework_destination}/Versions/B"
    sign_code_object "${sparkle_version_directory}/Autoupdate"
    sign_code_object "${sparkle_version_directory}/Updater.app"
    sign_code_object "${sparkle_version_directory}/XPCServices/Downloader.xpc"
    sign_code_object "${sparkle_version_directory}/XPCServices/Installer.xpc"
    sign_code_object "$sparkle_framework_destination"

    printf '  signing Astronomical executables...\n'
    # Leaf-first signing keeps every executable attributable before the app seal is created.
    for bundled_executable_name in astronomical-menu astronomicald astronomical-inference-worker; do
        sign_code_object "${app_bundle_path}/Contents/MacOS/${bundled_executable_name}"
    done

    printf '  signing app bundle...\n'
    plutil -lint "${app_bundle_path}/Contents/Info.plist"
    sign_code_object "$app_bundle_path"
    codesign --verify --deep --strict "$app_bundle_path"
    finish_phase "success"

    # ── Phase 5: Post-build validation ───────────────────────────────────

    start_phase 5 "post-build validation for isolated ${APPLICATION_CHANNEL} instance"
    validate_script="${repository_root}/scripts/internal/validate-macos-app.sh"
    if [ -x "$validate_script" ]; then
        validation_exit_code=0
        if [ "$APPLICATION_CHANNEL" = "stable" ]; then
            "$validate_script" --app-bundle "$app_bundle_path" --bundle-only || validation_exit_code=$?
        else
            "$validate_script" --app-bundle "$app_bundle_path" || validation_exit_code=$?
        fi
        if [ "$validation_exit_code" -eq 0 ]; then
            finish_phase "success"
        else
            finish_phase "failed"
            print_error "post-build validation failed (exit code ${validation_exit_code})"
            exit 1
        fi
    else
        print_error "validation script not found or not executable: ${validate_script}"
        print_error "skipping post-build validation"
        finish_phase "skipped"
    fi

    # ── Summary ──────────────────────────────────────────────────────────

    script_finish_seconds="$(date +%s)"
    total_elapsed_seconds=$((script_finish_seconds - script_start_seconds))

    printf '%s\n' ""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' "  Astronomical ${APPLICATION_CHANNEL} app ready in ${total_elapsed_seconds}s"
    printf '%s\n' "  ${app_bundle_path}"
    printf '%s\n' ""
    printf '%s\n' "  Launch: open \"${app_bundle_path}\""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' ""
}

main "$@"
