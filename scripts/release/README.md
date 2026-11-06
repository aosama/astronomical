# Stable Release Operations

The full release procedure is the `release-stable` skill at `.claude/skills/release-stable/SKILL.md`. This page lists the commands it uses.

Every release step is a typed `cargo xtask release <step>` command; ordinary development builds, commits, pushes, and pull requests must not invoke them. CI exercises their planning and checking functions through `cargo test -p xtask`, the typed packaging contracts through `cargo test-app-builder`, and the icon renderer through `swift scripts/internal/render-macos-app-icon-contract.swift`.

- Build Development with `cargo xtask app dev`.
- Verify ordinary changes with `cargo fmt --all -- --check` and `cargo test-integration-and-rest`.
- Build a Stable candidate with `cargo xtask app package --channel stable`.
- Run the packaging contracts explicitly with `cargo test -p xtask`, `cargo test-app-builder`, and `swift scripts/internal/render-macos-app-icon-contract.swift`.
- Prepare or publish a signed release only through `cargo xtask release prepare-and-publish --tag vX.Y.Z ...` (add `--publish` with the reviewed `--output-directory` to publish).
- The individual steps are `cargo xtask release create-dmg`, `validate-dmg`, `validate-distribution-app`, `notarize-dmg [--timeout-seconds N]`, `install-stable`, `build-and-install-stable`, `app-store-pkg`, and `accept-sparkle-update` (the end-to-end Sparkle update acceptance, using the fixtures in `fixtures/`).
- `cargo xtask release prepare-signed --tag vX.Y.Z [--notes-file PATH] [--output-directory PATH] [--publish]` is the convenience front door for the same publisher: it resolves the documented credential map (exactly one Developer ID identity, its Team ID, the Astronomical Notarization profile) and runs the publisher in the same process. It performs no artifact work itself.

Release commands may open Finder, mount disk images, access signing identities, submit notarization requests, or mutate GitHub release state. Review their arguments and prerequisites before execution.

Release preparation automatically keeps macOS `caffeinate` active for the preparation process so an idle screen lock cannot make a valid Data Protection Keychain notarization profile appear missing. A deliberate manual lock can still deny profile access; unlock the Mac and retry if the command reports that the profile is inaccessible.
