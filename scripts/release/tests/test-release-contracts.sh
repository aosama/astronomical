#!/usr/bin/env sh

# Runs the bounded user-journey contracts for macOS packaging and publication.

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"

for contract_test in \
    test-render-app-icon.sh \
    test-build-macos-app.sh \
    test-create-dmg.sh \
    test-validate-distribution-app.sh \
    test-notarize-dmg.sh \
    test-validate-dmg.sh \
    test-prepare-and-publish.sh \
    test-build-and-install-stable-app.sh \
    test-install-stable-app.sh
do
    printf '%s\n' "[release-contracts] test=${contract_test} status=start"
    "${repository_root}/scripts/release/tests/${contract_test}"
    printf '%s\n' "[release-contracts] test=${contract_test} status=success"
done
