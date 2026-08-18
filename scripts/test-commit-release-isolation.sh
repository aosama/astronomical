#!/usr/bin/env sh

# Guards the user journey where ordinary commit verification must never enter
# Stable installation, disk-image, notarization, Finder, or publication flows.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
verification_script="${repository_root}/scripts/verify-before-commit.sh"
verification_script_text="$(LC_ALL=C tr '\n' ' ' < "$verification_script")"

case "$verification_script_text" in
    *"scripts/release/"*|*"release/tests"*)
        print_error "ordinary commit verification references the release-only namespace"
        exit 1
        ;;
esac

for release_entry_point in \
    build-stable-app.sh \
    build-and-install-stable-app.sh \
    install-stable-app.sh \
    create-dmg.sh \
    notarize-dmg.sh \
    validate-dmg.sh \
    prepare-and-publish.sh
do
    [ -x "${repository_root}/scripts/release/${release_entry_point}" ] || {
        print_error "release entry point is unavailable: scripts/release/${release_entry_point}"
        exit 1
    }
done

if "${repository_root}/scripts/build-development-app.sh" --channel stable >/dev/null 2>&1; then
    print_error "the Development builder accepted Stable channel selection"
    exit 1
fi

printf '%s\n' "[commit-release-isolation] status=success"
