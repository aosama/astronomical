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
APPLICATION_CHANNEL="development"

# ── Helpers ────────────────────────────────────────────────────────────

script_start_seconds=0
current_phase=0

print_usage() {
    printf '%s\n' "Usage: scripts/make-astronomical-app.sh [--channel development|stable]"
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

main() {
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
    require_command swift
    require_command sysctl
    require_command codesign
    require_command plutil
    require_command iconutil
    require_command git
    require_command jq
    finish_phase "success"

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
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
    application_version="$(cargo metadata --no-deps --format-version 1 | jq --raw-output '.packages[] | select(.name == "astronomical-supervisor") | .version')"
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
    export CARGO_BUILD_JOBS="$logical_cpu_count"
    if command -v sccache >/dev/null 2>&1; then
        export RUSTC_WRAPPER=sccache
        printf '  using sccache wrapper, %s parallel jobs\n' "$logical_cpu_count"
    else
        printf '  %s parallel jobs (no sccache detected)\n' "$logical_cpu_count"
    fi
    # Cargo prints its own live compilation progress to stderr.
    cargo build --release \
        -p astronomical-inference-worker --bin astronomical-inference-worker \
        -p astronomical-supervisor --bin astronomicald
    swift build --configuration release --package-path "${repository_root}/apps/astronomical-menu"
    finish_phase "success"

    # ── Phase 4: Assemble and sign app bundle ────────────────────────────

    start_phase 4 "assemble and codesign app bundle"
    remove_previous_app_bundle "$app_bundle_path"
    mkdir -p "${app_bundle_path}/Contents/MacOS"
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
        printf '%s\n' '  <key>CFBundleIconFile</key><string>Astronomical.icns</string>'
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
    cp "${repository_root}/target/release/astronomicald" \
       "${app_bundle_path}/Contents/MacOS/astronomicald"
    cp "${repository_root}/target/release/astronomical-inference-worker" \
       "${app_bundle_path}/Contents/MacOS/astronomical-inference-worker"
    cp "${repository_root}/LICENSE" \
       "${app_bundle_path}/Contents/Resources/LICENSE"
    cp "${repository_root}/third-party/THIRD_PARTY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/THIRD_PARTY_NOTICES"
    cp "${repository_root}/third-party/RUST_DEPENDENCY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/RUST_DEPENDENCY_NOTICES"

    # Render the complete icon family from build identity rather than storing
    # release-specific artwork that can drift from the packaged version.
    iconset_directory="${app_bundle_path}/Contents/Resources/Astronomical.iconset"
    icon_resource="${app_bundle_path}/Contents/Resources/Astronomical.icns"
    swift "${repository_root}/scripts/render-astronomical-app-icon.swift" \
        --output-directory "$iconset_directory" \
        --version "$application_version" \
        --build-date "$build_date_utc"
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

    printf '  signing embedded executables...\n'
    # The macOS runtime signature monitor can reject linker-signed nested
    # executables even when static bundle verification succeeds. Sign each
    # executable before sealing the enclosing bundle.
    for bundled_executable_name in astronomical-menu astronomicald astronomical-inference-worker; do
        codesign --force --sign - "${app_bundle_path}/Contents/MacOS/${bundled_executable_name}"
    done

    printf '  signing app bundle...\n'
    plutil -lint "${app_bundle_path}/Contents/Info.plist"
    codesign --force --sign - "$app_bundle_path"
    codesign --verify --deep --strict "$app_bundle_path"
    finish_phase "success"

    # ── Phase 5: Post-build validation ───────────────────────────────────

    start_phase 5 "post-build validation for isolated ${APPLICATION_CHANNEL} instance"
    validate_script="${repository_root}/scripts/validate-astronomical-app.sh"
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
