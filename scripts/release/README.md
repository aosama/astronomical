# Stable Release Operations

Everything in this directory is release-only. Ordinary development builds, commits, pushes, and pull requests must not invoke these scripts.

- Build Development with `scripts/build-development-app.sh`.
- Verify ordinary changes with `scripts/verify-before-commit.sh`.
- Build a Stable candidate with `scripts/release/build-stable-app.sh`.
- Run all Stable packaging and publication contracts explicitly with `scripts/release/tests/test-release-contracts.sh`.
- Prepare or publish a signed release only through `scripts/release/prepare-and-publish.sh`.

Release scripts may open Finder, mount disk images, access signing identities, submit notarization requests, or mutate GitHub release state. Review their arguments and prerequisites before execution.
