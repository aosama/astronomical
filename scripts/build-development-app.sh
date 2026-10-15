#!/usr/bin/env sh

# Public entry point for building the isolated Development application.
# Stable selection and distribution credentials intentionally do not cross this boundary.
#
# The optimized build's live validation refuses to run while a development
# instance occupies the endpoint (it will not stop an instance it does not
# own), so a running development instance is stopped first and brought back
# after the build with the caller's environment intact - exported
# ASTRONOMICAL_* experiment flags survive the rebuild.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 0 ]; then
    print_error "scripts/build-development-app.sh does not accept release or channel arguments"
    exit 2
fi

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
dev_instance_script="${repository_root}/scripts/development/astronomical-dev-instance.sh"

development_instance_was_running=0
if [ -x "$dev_instance_script" ] && pgrep -f "astronomicald --instance development" > /dev/null 2>&1; then
    "$dev_instance_script" stop || true
    development_instance_was_running=1
fi

build_exit_code=0
"${repository_root}/scripts/internal/build-macos-app.sh" --channel development || build_exit_code=$?

if [ "$development_instance_was_running" -eq 1 ]; then
    "$dev_instance_script" start || print_error "could not restart the development instance automatically; start it with scripts/development/astronomical-dev-instance.sh start"
fi

exit "$build_exit_code"
