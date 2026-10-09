---
description: Prepare, sign, notarize, publish, and activate one Astronomical Stable release.
agent: build
subtask: false
---

# Release Astronomical Stable

Release version `$1`. If it is empty, ask the user for a numeric `MAJOR.MINOR.PATCH`; never infer it. Then use the confirmed value as `VERSION` and `TAG="v${VERSION}"`, and require it to be newer than the latest semantic release tag. This command authorizes the release work, but you must stop once after local artifact review and obtain the user's explicit approval before pushing the tag or publishing.

## Non-negotiable rules

- Work from the repository root and follow `AGENTS.md` and `repo-discovery-guide-for-agents.md`.
- Never use chat history, chat-memex, memory, or guesswork for credential names.
- Never commit or place credential values, local paths, Keychain output, or machine logs in GitHub content or the final report. Do not suppress authoritative release-script output merely because it contains local operator paths.
- Use only `scripts/release/prepare-and-publish.sh` to prepare or publish distribution artifacts.
- Never manually edit `site/appcast.xml`; it is a signed generated artifact.
- Never move or replace a pushed release tag. Fix forward with a new patch version.
- Do not promote the real release into `~/Applications` or launch it. Temporary mounting and copying by `validate-dmg.sh`, and installation and launch of the fictional fixture under `target/` by `accept-sparkle-update.sh`, are required isolated validation—not installation of the release. The developer must later receive the real app through the same public download or Sparkle update journey as every user.
- Every test must use its built-in timeout of at most 120 seconds. The notarization operation is not a test and uses the release script's 1,200-second bound.
- Keep commands live and unfiltered. Never pipe long-running commands through `head`, `tail`, or another buffering filter.

## Credential map

Use these sources exactly; no other credential discovery is allowed:

- **Developer ID identity:** the single Keychain code-signing identity whose quoted name starts with `Developer ID Application:`.
- **Team ID:** the 10-character value in the final parentheses of that identity.
- **Notarization profile:** `Astronomical Notarization`.
- **Sparkle private key:** the Keychain key automatically selected by Sparkle's `generate_appcast`. `scripts/release/accept-sparkle-update.sh` proves that it matches Astronomical's embedded public key.

Resolve the non-secret signing arguments without displaying them:

```sh
SIGNING_IDENTITY="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p')"
IDENTITY_COUNT="$(printf '%s\n' "$SIGNING_IDENTITY" | awk 'NF { count += 1 } END { print count + 0 }')"
[ "$IDENTITY_COUNT" = 1 ] || { printf '%s\n' 'Error: exactly one Developer ID Application identity is required' >&2; exit 1; }
TEAM_ID="$(printf '%s\n' "$SIGNING_IDENTITY" | sed -n 's/.*(\([[:alnum:]]\{10\}\))$/\1/p')"
[ "${#TEAM_ID}" = 10 ] || { printf '%s\n' 'Error: Developer ID identity does not contain a 10-character Team ID' >&2; exit 1; }
NOTARY_PROFILE="Astronomical Notarization"
xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null
```

If any credential check fails, stop and ask the user to unlock the Mac or repair the Keychain. Do not search chats for alternatives.

## 1. Establish scope and issue

1. Fetch remote state; require clean synchronized `main`, successful latest `main` CI, no existing `$TAG`, and no existing GitHub release for `$TAG`.
2. Set `PREVIOUS_TAG` to the highest existing semantic release tag and review `git log` and `git diff` from it to `main`.
3. Create one public issue titled `Release Astronomical ${VERSION}`. State the exact user-visible changes, compatibility, signing requirements, digest-bound publication, signed-feed activation, and public verification. Do not include private environment details.
4. Save the returned issue number as `ISSUE_NUMBER`. Both release pull requests must link only this issue.

## 2. Merge the version bump

1. Create branch `release/v${VERSION}`.
2. Change only `[workspace.package].version` in `Cargo.toml` to `$VERSION`.
3. Run `cargo update --workspace`; verify `Cargo.lock` changes only Astronomical workspace package versions.
4. Create `RELEASE_NOTES_FILE` with `mktemp "${TMPDIR:-/tmp}/astronomical-${TAG}-notes.XXXXXX.md"`, record the resulting path, and write concise public release notes there. Repeat that literal path in later tool calls because shell variables may not persist between calls. Include only changes proven by `PREVIOUS_TAG..HEAD`, compatibility, preserved user state, and the signed/notarized distribution statement. Keep this path out of public content.
5. Run, in order:

```sh
tests/scripts/release/test-release-contracts.sh
cargo fmt --all -- --check
cargo test-hermetic-and-rest
```

6. Inspect `git status`, `git diff`, and `git log --oneline -10`; commit only `Cargo.toml` and `Cargo.lock` with `chore: bump version to ${VERSION}`.
7. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}`. Wait for required checks, merge it, switch to `main`, pull with `--ff-only`, and verify local `HEAD`, `refs/remotes/origin/main`, and GitHub `main` are identical.

## 3. Preflight and prepare locally

1. Re-resolve the credential map above and run:

```sh
scripts/release/accept-sparkle-update.sh
```

Its transcript is long, so capture the exit code explicitly (`scripts/release/accept-sparkle-update.sh; echo EXIT=$?`); the terminal result otherwise scrolls out of reach.

2. Require a clean worktree, workspace version `$VERSION`, and release notes that are nonempty and public-safe. Re-resolve signing arguments in the same shell invocation that runs preparation; do not assume shell variables persisted from preflight.
3. Create a local lightweight tag with `git tag "$TAG"`. Do not push it until the publication approval gate passes.
4. Prepare the release:

```sh
scripts/release/prepare-and-publish.sh \
  --tag "$TAG" \
  --notes-file "$RELEASE_NOTES_FILE" \
  --signing-identity "$SIGNING_IDENTITY" \
  --team-id "$TEAM_ID" \
  --notary-profile "$NOTARY_PROFILE"
```

This authoritative script signs nested Sparkle code and Astronomical executables inside-out, signs the app with Hardened Runtime and timestamps, creates and signs the DMG, submits it to Apple, staples and validates the ticket, validates the mounted installation layout, Sparkle-signs the final stapled bytes, signs the feed, and writes `target/releases/${TAG}/release-manifest.json`. This output is long and carries the notarization submission ID that section 4 requires; if you background the run, keep its log so that ID and the final `Prepared signed release` line stay retrievable.

## 4. Mandatory publication approval gate

Validate `target/releases/${TAG}` without changing it:

- Manifest tag, version, and commit equal `$TAG`, `$VERSION`, and `HEAD`.
- DMG, notes, and appcast sizes and SHA-256 digests equal the manifest.
- The DMG name is `Astronomical-${VERSION}-macOS-arm64.dmg`.
- `scripts/release/validate-dmg.sh --dmg "target/releases/${TAG}/Astronomical-${VERSION}-macOS-arm64.dmg"` passes for that exact DMG. It accepts `--dmg`, not a positional path.
- Record the notarization submission ID from preparation, query it with `xcrun notarytool log SUBMISSION_ID --keychain-profile "Astronomical Notarization"`, and require `Accepted` with zero issues. Parse only status and issue count; do not publish the complete log.
- Appcast XML is valid and contains `$VERSION`, a build number greater than the previous release, macOS `14.0` and above, `arm64`, the exact GitHub release URL, an enclosure Ed25519 signature, and a signed-feed envelope.
- The prepared directory contains exactly the DMG, copied Markdown notes, `appcast.xml`, and `release-manifest.json`.

Report only the version, commit, artifact names, sizes, digests, and validation outcomes. Ask the user to approve publication. Do not push the tag or mutate GitHub release state until approved.

## 5. Publish the reviewed bytes

After approval, push the already-reviewed immutable tag first; the publish preflight requires that the remote tag already resolves to the prepared commit:

```sh
git push origin "refs/tags/${TAG}"
scripts/release/prepare-and-publish.sh \
  --tag "$TAG" \
  --output-directory "target/releases/${TAG}" \
  --publish
```

Use the exact option name `--output-directory`; `--output-dir` is not accepted. If publication is interrupted after a matching draft release exists, rerun this same command so the authoritative script resumes the draft rather than creating a second release. Require the remote tag to resolve to the manifest commit. Verify the public GitHub release is Stable, not draft or prerelease, has the exact title and notes, and contains exactly the manifest-named DMG with matching size and SHA-256 digest. Download the public DMG to a temporary directory, verify its digest against the manifest, then delete the temporary download. Do not mount, install, or launch it.

Comparing the published notes to `$RELEASE_NOTES_FILE` can show one spurious trailing blank line, because `gh release view --json body --jq .body` appends a newline to what it prints; treat only that trailing blank line as equal. A HEAD probe of the enclosure URL likewise returns an empty body by design, so use the HTTP status for reachability and the full download for the digest.

Publication intentionally leaves the old public appcast active and writes the new signed `site/appcast.xml` as an unstaged worktree modification.

## 6. Activate the signed appcast

1. Create branch `release/v${VERSION}-appcast` without discarding the appcast worktree modification.
2. Require `cmp site/appcast.xml target/releases/${TAG}/appcast.xml` to succeed. Do not format or edit the XML.
3. Run:

```sh
scripts/release/accept-sparkle-update.sh
cargo fmt --all -- --check
cargo test-hermetic-and-rest
```

4. Inspect status, diff, and recent log; commit only `site/appcast.xml` with `release: activate v${VERSION} update feed`.
5. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}`. This appcast-only change is maintenance work; `Fixes` is reserved for implementation work. Wait for required checks and merge it. The macOS hermetic verification job reports `skipping` rather than `pass`, which is expected for a signed site artifact; the other required checks must still be terminal. The exact heading matters: placing `Refs #N` elsewhere in the body fails the repository validator.
6. Synchronize local `main` with the squash-merged remote commit without rebasing. A clean local branch can diverge by one commit after the remote squash merge; fetch `origin/main`, then use a fast-forward or clean reset to the remote branch. Wait for the Pages workflow for the merge commit, and require success.
7. Download the public `appcast.xml` to a temporary file. Require it to be byte-identical to `target/releases/${TAG}/appcast.xml` and to match the manifest digest. Verify its enclosure URL is reachable and the remote DMG still matches the manifest. Delete temporary files.

## 7. Finish

Require all of the following before reporting success:

- GitHub release `$TAG` is public and is the latest Stable release.
- Public Pages appcast is the exact prepared signed appcast.
- Release issue is explicitly closed after the appcast PR merge, Pages deployment, and public-byte verification. A `Refs #N` maintenance PR does not close it automatically.
- Local `main` equals remote `main` and the tracked worktree is clean.
- The ignored `target/releases/${TAG}` evidence directory remains intact.
- The real Astronomical release was not installed or launched locally; isolated temporary validation fixtures are allowed as described above.

Tell the user that Astronomical `$VERSION` is available through the normal public release and Sparkle channels. Do not acquire it for them; they will receive it through the ordinary user journey.

## Failure rules

- Before tag push: fix the cause, remove only an invalid `target/releases/${TAG}` after explicit user approval, recreate the local tag only if its commit is wrong, and prepare again.
- After tag push: never move the tag; withhold appcast activation and fix forward with a new patch release.
- Interrupted matching draft publication: rerun the same `--publish` command; the script safely resumes it.
- A GitHub HTTP 404 from `verify-remote-release-tag` means the reviewed tag has not been pushed yet; push that exact local tag, never recreate or move it.
- Conflicting remote release, asset, notes, digest, or tag: stop without clobbering anything.
- Failed Pages deployment or public-byte comparison: keep the prior feed active when possible, do not declare success, and repair through a pull request.

## Known gotchas (from the 0.2.58 release run)

- `accept-sparkle-update.sh` rebuilds Sparkle's `sparkle-cli` with `xcodebuild` only when the prebuilt binary is missing. A refreshed checkout (wiped `build/` directory) therefore exposes a latent failure: Sparkle's Xcode project pins `MACOSX_DEPLOYMENT_TARGET=10.13`, which is below the minimum supported by current Xcode SDKs, and the build fails with exit 65. The script overrides the deployment target on the command line for this fixture-only build; release bytes are unaffected because the app and `generate_appcast` come from SwiftPM artifacts. If the acceptance preflight still fails at `resolve-sparkle-tools`, check for a new SDK minimum and bump the override in `build_sparkle_cli`.
- Release notes created through `mktemp "${TMPDIR:-/tmp}/name.XXXXXX.md"` can end up with a literal, unreplaced `XXXXXX` in the filename. The file is still valid; use the printed path verbatim everywhere (manifest, `--notes-file`) instead of assuming a clean temp name.
- The PR issue compliance validator requires the `## Linked issue` section to contain exactly one reference line (`Fixes|Closes|Resolves|Refs #N`). Any prose inside that section fails validation; put narrative above the heading. For release maintenance PRs (appcast activation, version bump) use `Refs #N`; `Fixes` is reserved for implementation work.
- Expected CI duration for hermetic verification PRs is roughly 7-8 minutes; do not treat a pending job at the 2-3 minute mark as stuck.
- `scripts/release/prepare-and-publish.sh` requires signing identity, team ID, and notary profile resolved inside the same shell invocation as the script call; resolving them in an earlier command does not persist.

## FAQ

- Why did the Sparkle acceptance fail right after it worked in a previous release? The acceptance script skips `xcodebuild` when a prebuilt `sparkle-cli` exists. A prior release can succeed with the stale binary; the next checkout refresh deletes `build/`, forces the rebuild, and only then exposes the deployment-target incompatibility.
- Why is the macOS hermetic verification check "skipping" on the appcast PR? The classify workflow marks signed site-artifact-only changes as not requiring the full hermetic run. The remaining required checks must still reach a terminal success state.
- Why does the `## Linked issue` section reject my explanation? The repository validator enforces exactly one machine-parsable reference line in that section so release provenance stays auditable. Explanations belong above the heading.
- Can I rerun a partially completed publish? Yes. Rerun the same `--publish` command with `--output-directory` pointing at the prepared directory; the script resumes the matching draft publication safely.
- Why do verification steps run plain commands instead of a gate script? The commit-gate wrapper was retired (issue #983) in favor of the plain commands it wrapped: `cargo fmt --all -- --check` and `cargo test-hermetic-and-rest`. The Stable packaging contracts run explicitly through `tests/scripts/release/test-release-contracts.sh`, relocated from `scripts/release/tests/`.
