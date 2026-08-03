#!/usr/bin/env sh

# Builds optimized release binaries and assembles an Astronomical.app bundle
# in target/astronomical-macos-release/, then runs post-build validation.
#
# Exit codes:
#   0 — build and validation succeeded
#   1 — build or validation failure
#
# Every phase reports its number, start, end, elapsed time, and ETA.

set -eu

# ── Constants ──────────────────────────────────────────────────────────

readonly TOTAL_PHASES=5

# ── Helpers ────────────────────────────────────────────────────────────

script_start_seconds=0
current_phase=0

print_usage() {
    printf '%s\n' "Usage: scripts/make-astronomical-app.sh"
    printf '%s\n' ""
    printf '%s\n' "Builds release binaries, assembles Astronomical.app, and runs"
    printf '%s\n' "post-build validation. Output:"
    printf '%s\n' "  target/astronomical-macos-release/Astronomical.app"
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
    expected_suffix="/target/astronomical-macos-release/Astronomical.app"

    case "$bundle_path" in
        *"$expected_suffix")
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
    printf '%s\n' "  Astronomical.app Build Pipeline (${TOTAL_PHASES} phases)"
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
    finish_phase "success"

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    release_directory="${repository_root}/target/astronomical-macos-release"
    app_bundle_path="${release_directory}/Astronomical.app"

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
        -p astronomical-supervisor --bin astronomicald \
        -p astronomical-model-serving --features direct-mlx --bin astronomical-model-preparer
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
        printf '%s\n' '  <key>CFBundleName</key><string>Astronomical</string>'
        printf '%s\n' '  <key>CFBundleDisplayName</key><string>Astronomical</string>'
        printf '%s\n' '  <key>CFBundleExecutable</key><string>astronomical-menu</string>'
        printf '%s\n' '  <key>CFBundleIdentifier</key><string>dev.astronomical.app</string>'
        printf '%s\n' '  <key>CFBundleVersion</key><string>0.1.0</string>'
        printf '%s\n' '  <key>CFBundleShortVersionString</key><string>0.1.0</string>'
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
    cp "${repository_root}/target/release/astronomical-model-preparer" \
       "${app_bundle_path}/Contents/MacOS/astronomical-model-preparer"
    cp "${repository_root}/LICENSE" \
       "${app_bundle_path}/Contents/Resources/LICENSE"
    cp "${repository_root}/third-party/THIRD_PARTY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/THIRD_PARTY_NOTICES"
    cp "${repository_root}/third-party/RUST_DEPENDENCY_NOTICES" \
       "${app_bundle_path}/Contents/Resources/RUST_DEPENDENCY_NOTICES"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomical-menu"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomicald"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomical-inference-worker"
    chmod +x "${app_bundle_path}/Contents/MacOS/astronomical-model-preparer"

    printf '  signing embedded executables...\n'
    # The macOS runtime signature monitor can reject linker-signed nested
    # executables even when static bundle verification succeeds. Sign each
    # executable before sealing the enclosing bundle.
    for bundled_executable_name in astronomical-menu astronomicald astronomical-inference-worker astronomical-model-preparer; do
        codesign --force --sign - "${app_bundle_path}/Contents/MacOS/${bundled_executable_name}"
    done

    printf '  signing app bundle...\n'
    plutil -lint "${app_bundle_path}/Contents/Info.plist"
    codesign --force --sign - "$app_bundle_path"
    codesign --verify --deep --strict "$app_bundle_path"
    finish_phase "success"

    # ── Phase 5: Post-build validation ───────────────────────────────────

    start_phase 5 "post-build validation (health, models, chat completion)"
    validate_script="${repository_root}/scripts/validate-astronomical-app.sh"
    if [ -x "$validate_script" ]; then
        if "$validate_script" --app-bundle "$app_bundle_path"; then
            finish_phase "success"
        else
            validate_exit_code=$?
            finish_phase "failed"
            print_error "post-build validation failed (exit code ${validate_exit_code})"
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
    printf '%s\n' "  Astronomical.app ready in ${total_elapsed_seconds}s"
    printf '%s\n' "  ${app_bundle_path}"
    printf '%s\n' ""
    printf '%s\n' "  Launch: open \"${app_bundle_path}\""
    printf '%s\n' "══════════════════════════════════════════════════════════════"
    printf '%s\n' ""
}

main "$@"
