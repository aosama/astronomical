#!/usr/bin/env sh

# Runs the bounded user-journey contracts for macOS packaging and publication.

set -eu

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"

for contract_test in \
    test-render-astronomical-app-icon.sh \
    test-make-astronomical-app-output.sh \
    test-make-astronomical-dmg.sh \
    test-validate-astronomical-distribution-app.sh \
    test-notarize-astronomical-dmg.sh \
    test-validate-astronomical-dmg.sh \
    test-publish-astronomical-release.sh
do
    printf '%s\n' "[macos-distribution-contracts] test=${contract_test} status=start"
    "${repository_root}/scripts/${contract_test}"
    printf '%s\n' "[macos-distribution-contracts] test=${contract_test} status=success"
done
