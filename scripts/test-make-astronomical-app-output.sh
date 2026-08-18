#!/usr/bin/env sh

# Verifies that generated app bundles remain available beneath Spotlight-excluded build directories.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe builder test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

write_successful_command() {
    command_path="$1"
    cat > "$command_path" <<'COMMAND'
#!/usr/bin/env sh
exit 0
COMMAND
    chmod +x "$command_path"
}

assert_bundle_exists() {
    expected_app_bundle="$1"
    bundle_information_plist="${expected_app_bundle}/Contents/Info.plist"
    [ -x "${expected_app_bundle}/Contents/MacOS/astronomical-menu" ] || {
        print_error "menu executable is unavailable in ${expected_app_bundle}"
        exit 1
    }
    [ -x "${expected_app_bundle}/Contents/MacOS/astronomicald" ] || {
        print_error "daemon executable is unavailable in ${expected_app_bundle}"
        exit 1
    }
    [ -x "${expected_app_bundle}/Contents/MacOS/astronomical-inference-worker" ] || {
        print_error "worker executable is unavailable in ${expected_app_bundle}"
        exit 1
    }
    [ -s "${expected_app_bundle}/Contents/Resources/Astronomical.icns" ] || {
        print_error "packaged macOS icon is unavailable in ${expected_app_bundle}"
        exit 1
    }
    [ -s "${expected_app_bundle}/Contents/Resources/SPARKLE_LICENSE" ] || {
        print_error "Sparkle license is unavailable in ${expected_app_bundle}"
        exit 1
    }
    [ -d "${expected_app_bundle}/Contents/Frameworks/Sparkle.framework" ] || {
        print_error "packaged Sparkle framework is unavailable in ${expected_app_bundle}"
        exit 1
    }
    grep -F '<key>CFBundleIconFile</key><string>Astronomical.icns</string>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not select Astronomical.icns"
            exit 1
        }
    grep -F '<key>AstronomicalBuildDate</key><string>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not include the icon build date"
            exit 1
        }
    grep -F '<key>SUFeedURL</key><string>https://example.github.io/astronomical/appcast.xml</string>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not select the GitHub Pages update feed"
            exit 1
        }
    grep -F '<key>SUPublicEDKey</key><string>g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ=</string>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not embed the Sparkle public key"
            exit 1
        }
    grep -F '<key>SUVerifyUpdateBeforeExtraction</key><true/>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not require update verification before extraction"
            exit 1
        }
    grep -F '<key>SURequireSignedFeed</key><true/>' "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not require a signed update feed"
            exit 1
        }
    grep -F '<key>SUEnableAutomaticChecks</key><true/>' "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not enable automatic update checks by default"
            exit 1
        }
    grep -F '<key>SUScheduledCheckInterval</key><integer>86400</integer>' \
        "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not bound automatic checks to once per day"
            exit 1
        }
    grep -F '<key>SUAutomaticallyUpdate</key><false/>' "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata unexpectedly enables automatic update installation"
            exit 1
        }
    grep -F '<key>SUEnableSystemProfiling</key><false/>' "$bundle_information_plist" >/dev/null || {
            print_error "bundle metadata does not disable Sparkle system profiling"
            exit 1
        }
}

main() {
    for required_command in timeout mktemp grep; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-app-builder.XXXXXX")"
    sandbox_repository="${SANDBOX_DIRECTORY}/repository"
    sandbox_scripts_directory="${sandbox_repository}/scripts"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    mkdir -p "$sandbox_scripts_directory" "$fake_command_directory" \
        "${sandbox_repository}/apps/astronomical-menu/.build/release/Sparkle.framework" \
        "${sandbox_repository}/apps/astronomical-menu/.build/checkouts/Sparkle" \
        "${sandbox_repository}/third-party"
    cp "${repository_root}/scripts/make-astronomical-app.sh" \
        "${sandbox_scripts_directory}/make-astronomical-app.sh"
    chmod +x "${sandbox_scripts_directory}/make-astronomical-app.sh"
    printf '%s\n' fixture > "${sandbox_repository}/LICENSE"
    printf '%s\n' fixture > "${sandbox_repository}/third-party/THIRD_PARTY_NOTICES"
    printf '%s\n' fixture > "${sandbox_repository}/third-party/RUST_DEPENDENCY_NOTICES"
    printf '%s\n' fixture > "${sandbox_repository}/apps/astronomical-menu/.build/checkouts/Sparkle/LICENSE"

    cat > "${fake_command_directory}/cargo" <<'CARGO'
#!/usr/bin/env sh
if [ "${1:-}" = "metadata" ]; then
    printf '%s\n' '{"packages":[{"name":"astronomical-supervisor","version":"0.2.0","repository":"https://github.com/example/astronomical"}]}'
    exit 0
fi
mkdir -p target/release
printf '%s\n' '#!/usr/bin/env sh' 'exit 0' > target/release/astronomicald
cp target/release/astronomicald target/release/astronomical-inference-worker
chmod +x target/release/astronomicald target/release/astronomical-inference-worker
CARGO
    cat > "${fake_command_directory}/swift" <<'SWIFT'
#!/usr/bin/env sh
if [ "${1:-}" != "build" ]; then
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--output-directory" ]; then
            iconset_directory="$2"
            break
        fi
        shift
    done
    mkdir -p "${iconset_directory:?iconset directory is required}"
    for icon_name in \
        icon_16x16.png icon_16x16@2x.png \
        icon_32x32.png icon_32x32@2x.png \
        icon_128x128.png icon_128x128@2x.png \
        icon_256x256.png icon_256x256@2x.png \
        icon_512x512.png icon_512x512@2x.png
    do
        printf '%s\n' fixture > "${iconset_directory}/${icon_name}"
    done
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--package-path" ]; then
        package_path="$2"
        break
    fi
    shift
done
mkdir -p "${package_path:?package path is required}/.build/release"
printf '%s\n' '#!/usr/bin/env sh' 'exit 0' > "${package_path}/.build/release/astronomical-menu"
chmod +x "${package_path}/.build/release/astronomical-menu"
SWIFT
    cat > "${fake_command_directory}/iconutil" <<'ICONUTIL'
#!/usr/bin/env sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then
        output_path="$2"
        break
    fi
    shift
done
printf '%s\n' fixture > "${output_path:?icon output path is required}"
ICONUTIL
    cat > "${fake_command_directory}/git" <<'GIT'
#!/usr/bin/env sh
case "${1:-}" in
    rev-parse) printf '%s\n' abcdef123456 ;;
    rev-list) printf '%s\n' 107 ;;
    status) exit 0 ;;
    *) exit 1 ;;
esac
GIT
    cat > "${fake_command_directory}/sysctl" <<'SYSCTL'
#!/usr/bin/env sh
printf '%s\n' 4
SYSCTL
    cat > "${fake_command_directory}/jq" <<'JQ'
#!/usr/bin/env sh
case "$*" in
    *'.version'*) printf '%s\n' '0.2.0' ;;
    *'.repository'*) printf '%s\n' 'https://github.com/example/astronomical' ;;
    *) exit 1 ;;
esac
JQ
    chmod +x "${fake_command_directory}"/*
    for command_name in cmake xcrun codesign install_name_tool plutil; do
        write_successful_command "${fake_command_directory}/${command_name}"
    done
    write_successful_command "${sandbox_scripts_directory}/bootstrap-native-dependencies.sh"
    write_successful_command "${sandbox_scripts_directory}/validate-astronomical-app.sh"

    printf '%s\n' '[app-builder-test] case=development-output-is-noindex status=start'
    (CDPATH='' cd -- "$sandbox_repository" && \
        PATH="${fake_command_directory}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            "${sandbox_scripts_directory}/make-astronomical-app.sh" --channel development)
    development_app_bundle="${sandbox_repository}/target/astronomical-macos-development.noindex/Astronomical Development.app"
    assert_bundle_exists "$development_app_bundle"
    printf '%s\n' '[app-builder-test] case=development-output-is-noindex status=success'

    printf '%s\n' '[app-builder-test] case=stable-output-is-noindex status=start'
    (CDPATH='' cd -- "$sandbox_repository" && \
        PATH="${fake_command_directory}:${PATH}" timeout "$SUBJECT_TIMEOUT_SECONDS" \
            "${sandbox_scripts_directory}/make-astronomical-app.sh" --channel stable)
    stable_app_bundle="${sandbox_repository}/target/astronomical-macos-stable.noindex/Astronomical.app"
    assert_bundle_exists "$stable_app_bundle"
    [ -d "$development_app_bundle" ] || {
        print_error "building Stable unexpectedly removed the Development build artifact"
        exit 1
    }
    printf '%s\n' '[app-builder-test] case=stable-output-is-noindex status=success'
}

main "$@"
