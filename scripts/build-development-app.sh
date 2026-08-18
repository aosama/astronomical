#!/usr/bin/env sh

# Public entry point for building the isolated Development application.
# Stable selection and distribution credentials intentionally do not cross this boundary.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 0 ]; then
    print_error "scripts/build-development-app.sh does not accept release or channel arguments"
    exit 2
fi

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
exec "${repository_root}/scripts/internal/build-macos-app.sh" --channel development
