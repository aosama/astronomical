#!/usr/bin/env sh

# Runs the bounded user-journey contracts for macOS packaging and publication.

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"

for contract_test in \
    test-render-app-icon.sh
do
    printf '%s\n' "[release-contracts] test=${contract_test} status=start"
    "${repository_root}/scripts/release/tests/${contract_test}"
    printf '%s\n' "[release-contracts] test=${contract_test} status=success"
done
