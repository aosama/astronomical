#!/usr/bin/env sh

# Builds, validates, and explicitly promotes one clean Stable app bundle.

set -eu

DRY_RUN="false"

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/release/build-and-install-stable-app.sh [--dry-run]"
    printf '%s\n' "Builds and validates Stable, then installs it into ~/Applications."
    printf '%s\n' "--dry-run builds Stable and previews the installation without promoting it."
}

run_step() {
    step_name="$1"
    shift
    step_started_at="$(date +%s)"
    printf '%s step=%s status=start\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name"
    if "$@"; then
        printf '%s step=%s status=success elapsed_seconds=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" \
            "$(( $(date +%s) - step_started_at ))"
        return
    else
        # Capture the delegated command status inside the else branch. POSIX
        # shells report the completed `if` statement itself as successful.
        step_exit_code=$?
    fi
    printf '%s step=%s status=failed elapsed_seconds=%s exit_code=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$step_name" \
        "$(( $(date +%s) - step_started_at ))" "$step_exit_code" >&2
    return "$step_exit_code"
}

main() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dry-run)
                [ "$DRY_RUN" = "false" ] || { print_error "--dry-run may be supplied only once"; exit 2; }
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

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"
    stable_builder="${repository_root}/scripts/release/build-stable-app.sh"
    stable_installer="${repository_root}/scripts/release/install-stable-app.sh"

    run_step "build-and-validate-stable" "$stable_builder"
    if [ "$DRY_RUN" = "true" ]; then
        run_step "preview-stable-installation" "$stable_installer" --dry-run
    else
        run_step "install-stable" "$stable_installer"
    fi
}

main "$@"
