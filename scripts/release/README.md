# Stable Release Operations

Everything in this directory is release-only: ordinary development builds, commits, pushes, and pull requests must not invoke these scripts. The exception is `tests/test-release-contracts.sh`, which CI runs on every pull request (the `macOS packaging release contracts` step) so the packaging journeys cannot rot between releases; the app-builder journey additionally runs as `cargo test-app-builder` on the macOS build job.

- Build Development with `cargo xtask app dev`.
- Verify ordinary changes with `cargo fmt --all -- --check` and `cargo test-integration-and-rest`.
- Build a Stable candidate with `cargo xtask app package --channel stable`.
- Run all Stable packaging and publication contracts explicitly with `scripts/release/tests/test-release-contracts.sh`.
- Prepare or publish a signed release only through `scripts/release/prepare-and-publish.sh`.
- The disk image steps are typed commands: `cargo xtask release create-dmg` and `cargo xtask release validate-dmg`; the remaining release steps are being ported the same way.
- `scripts/release/prepare-signed-release.sh` is the convenience front door for the same publisher: it resolves the documented credential map (single Developer ID identity, Team ID, notarization profile) in the caller's shell process and execs the publisher with `--tag`/`--notes-file` plus optional `--output-directory` and `--publish`. It performs no artifact work itself.

Release scripts may open Finder, mount disk images, access signing identities, submit notarization requests, or mutate GitHub release state. Review their arguments and prerequisites before execution.

Release preparation automatically keeps macOS `caffeinate` active for the preparation process so an idle screen lock cannot make a valid Data Protection Keychain notarization profile appear missing. A deliberate manual lock can still deny profile access; unlock the Mac and retry if the script reports that the profile is inaccessible.
