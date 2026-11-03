---
description: Prepare, sign, notarize, publish, and activate one Astronomical Stable release.
agent: build
subtask: false
---

# Release Astronomical Stable

Release version `$1`. If it is empty, ask the user for a numeric `MAJOR.MINOR.PATCH`; never infer it. Then use the confirmed value as `VERSION` and `TAG="v${VERSION}"`, and require it to be newer than the latest semantic release tag. This command authorizes the release work, but you must stop once after local artifact review and obtain the user's explicit approval before pushing the tag or publishing.

## Non-negotiable rules

- Work from the repository root and follow `AGENTS.md`.
- Never use chat history, chat-memex, memory, or guesswork for credential names.
- Never commit or place credential values, local paths, Keychain output, or machine logs in GitHub content or the final report. Do not suppress authoritative release-command output merely because it contains local operator paths.
- Use only `cargo xtask release prepare-and-publish` (or its credential-resolving front door `cargo xtask release prepare-signed`) to prepare or publish distribution artifacts.
- Never manually edit `site/appcast.xml`; it is a signed generated artifact.
- Never move or replace a pushed release tag. Fix forward with a new patch version.
- Do not promote the real release into `~/Applications` or launch it. Temporary mounting and copying by `cargo xtask release validate-dmg`, and installation and launch of the fictional fixture under `target/` by `cargo xtask release accept-sparkle-update`, are required isolated validation—not installation of the release. The developer must later receive the real app through the same public download or Sparkle update journey as every user.
- Every test must use its built-in timeout of at most 120 seconds. The Sparkle acceptance's 120-second budget covers the acceptance itself; resolving its Swift and Xcode tools beforehand reports its own progress. The notarization operation is not a test and uses `cargo xtask release notarize-dmg`'s 180-second default, overridable with `--timeout-seconds N`.
- Keep commands live and unfiltered. Never pipe long-running commands through `head`, `tail`, or another buffering filter.

## Credential map

Use these sources exactly; no other credential discovery is allowed:

- **Developer ID identity:** the single Keychain code-signing identity whose quoted name starts with `Developer ID Application:`.
- **Team ID:** the 10-character value in the final parentheses of that identity.
- **Notarization profile:** `Astronomical Notarization`.
- **Sparkle private key:** the Keychain key automatically selected by Sparkle's `generate_appcast`. `cargo xtask release accept-sparkle-update` proves that it matches Astronomical's embedded public key.

`cargo xtask release prepare-signed` resolves this map itself, failing closed unless exactly one Developer ID Application identity and a 10-character Team ID are found, and it checks the notarization profile is readable before any build starts. Use it instead of resolving the values by hand. To check the profile alone during preflight:

```sh
xcrun notarytool history --keychain-profile "Astronomical Notarization" >/dev/null
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
3. Run `cargo update --workspace`; verify `Cargo.lock` changes only Astronomical workspace package versions. The notices check honors text equality, so a version bump whose license output is unchanged passes with a behind digest memo (`digest_memo=behind` in the log); optionally run `cargo xtask notices` in the bump branch to keep later checks on the fast path instead of paying the cargo-about harvest.
4. Create `RELEASE_NOTES_FILE` with `mktemp "${TMPDIR:-/tmp}/astronomical-${TAG}-notes.md"`, record the resulting path, and write concise public release notes there. Repeat that literal path in later tool calls because shell variables may not persist between calls. Include only changes proven by `PREVIOUS_TAG..HEAD`, compatibility, preserved user state, and the signed/notarized distribution statement. Keep this path out of public content.
5. Run, in order:

```sh
swift scripts/internal/render-macos-app-icon-contract.swift
cargo test -p xtask
cargo test-app-builder
cargo fmt --all -- --check
cargo test-integration-and-rest
```

The packaging contracts are split three ways: the app icon renderer contract is a Swift program, the release steps' decisions (DMG layout, signatures, notarization plan, installation, publication) are `cargo test -p xtask` unit tests, and the typed `cargo xtask app package` contracts (signing plans, Cargo arguments, bundle layout, plist checks) run as the `cargo test-app-builder` lane, so a release run invokes all three.

6. Inspect `git status`, `git diff`, and `git log --oneline -10`; commit only `Cargo.toml`, `Cargo.lock`, and the regenerated `third-party/RUST_DEPENDENCY_NOTICES.digest` with `chore: bump version to ${VERSION}`.
7. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}`. Wait for required checks, merge it, then synchronize the release commit: in a primary checkout, switch to `main` and pull with `--ff-only`; in a linked worktree whose primary checkout owns `main`, fetch and check out the merge commit directly (`git checkout --detach origin/main`). Either way, verify local `HEAD`, `refs/remotes/origin/main`, and GitHub `main` resolve to the same commit.

## 3. Preflight and prepare locally

1. Check the notarization profile as described above and run:

```sh
cargo xtask release accept-sparkle-update
```

Its transcript is long, so capture the exit code explicitly (`cargo xtask release accept-sparkle-update; echo EXIT=$?`); the terminal result otherwise scrolls out of reach.

2. Run the installed-variant round trip — the gate that catches pack variants the single-pack journeys structurally miss (a real installed export whose config shape the loader refuses must fail here, before the release ships). First read `cargo xtask app instance status`: if the Development instance holds a model, ask the user before stopping it with `cargo xtask app instance stop`, because the round trip loads every installed model onto the same graphics processor. The test launches the real worker and daemon binaries from `target/release`, so build them first:

```sh
cargo build --release --locked -p astronomical-inference-worker -p astronomical-daemon --bins
cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving should_round_trip_every_installed_chat_model_variant -- --ignored --nocapture --test-threads=1
```

Every chat-capable model the Development configuration's model directories discover must load and stream a non-empty sampled reply (roughly one hundred tokens in at temperature 1.0, one-hundred-token budget, 120-second budget per model). The Development configuration is the single source of the variant inventory. A variant that fails to load or replies empty blocks the release; fix forward before tagging. This journey needs a real graphics processor and installed model inventory — it never runs in continuous integration.

3. Require a clean worktree, workspace version `$VERSION`, and release notes that are nonempty and public-safe.
4. Create a local lightweight tag with `git tag "$TAG"`. Do not push it until the publication approval gate passes.
5. Prepare the release:

```sh
cargo xtask release prepare-signed \
  --tag "$TAG" \
  --notes-file "$RELEASE_NOTES_FILE"
```

This authoritative command resolves the credential map, then signs nested Sparkle code and Astronomical executables inside-out, signs the app with Hardened Runtime and timestamps, creates and signs the DMG, submits it to Apple, staples and validates the ticket, validates the mounted installation layout, Sparkle-signs the final stapled bytes, signs the feed, and writes `target/releases/${TAG}/release-manifest.json`. This output is long and carries the notarization submission ID that section 4 requires; if you background the run, keep its log so that ID and the final `Prepared signed release` line stay retrievable.

## 4. Mandatory publication approval gate

Validate `target/releases/${TAG}` without changing it:

- Manifest tag, version, and commit equal `$TAG`, `$VERSION`, and `HEAD`.
- DMG, notes, and appcast sizes and SHA-256 digests equal the manifest.
- The DMG name is `Astronomical-${VERSION}-macOS-arm64.dmg`.
- `cargo xtask release validate-dmg --dmg "target/releases/${TAG}/Astronomical-${VERSION}-macOS-arm64.dmg"` passes for that exact DMG. It accepts `--dmg`, not a positional path.
- Record the notarization submission ID from preparation, query it with `xcrun notarytool log SUBMISSION_ID --keychain-profile "Astronomical Notarization"`, and require `Accepted` with zero issues. Parse only status and issue count; do not publish the complete log.
- Appcast XML is valid and contains `$VERSION`, a build number greater than the previous release, macOS `14.0` and above, `arm64`, the exact GitHub release URL, an enclosure Ed25519 signature, and a signed-feed envelope.
- The prepared directory contains exactly the DMG, copied Markdown notes, `appcast.xml`, and `release-manifest.json`.

Report only the version, commit, artifact names, sizes, digests, and validation outcomes. Ask the user to approve publication. Do not push the tag or mutate GitHub release state until approved.

## 5. Publish the reviewed bytes

After approval, push the already-reviewed immutable tag first; the publish preflight requires that the remote tag already resolves to the prepared commit:

```sh
git push origin "refs/tags/${TAG}"
cargo xtask release prepare-signed \
  --tag "$TAG" \
  --output-directory "target/releases/${TAG}" \
  --publish
```

Use the exact option name `--output-directory`; `--output-dir` is not accepted. If publication is interrupted after a matching draft release exists, rerun this same command so it resumes the draft rather than creating a second release. Require the remote tag to resolve to the manifest commit. Verify the public GitHub release is Stable, not draft or prerelease, has the exact title and notes, and contains exactly the manifest-named DMG with matching size and SHA-256 digest. Download the public DMG to a temporary directory, verify its digest against the manifest, then delete the temporary download. Do not mount, install, or launch it.

Comparing the published notes to `$RELEASE_NOTES_FILE` can show one spurious trailing blank line, because `gh release view --json body --jq .body` appends a newline to what it prints; treat only that trailing blank line as equal. A HEAD probe of the enclosure URL likewise returns an empty body by design, so use the HTTP status for reachability and the full download for the digest.

Publication intentionally leaves the old public appcast active and writes the new signed `site/appcast.xml` as an unstaged worktree modification.

## 6. Activate the signed appcast

1. Create branch `release/v${VERSION}-appcast` without discarding the appcast worktree modification.
2. Require `cmp site/appcast.xml target/releases/${TAG}/appcast.xml` to succeed. Do not format or edit the XML.
3. Run:

```sh
cargo xtask release accept-sparkle-update
cargo fmt --all -- --check
cargo test-integration-and-rest
```

4. Inspect status, diff, and recent log; commit only `site/appcast.xml` with `release: activate v${VERSION} update feed`.
5. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}`. This appcast-only change is maintenance work; `Fixes` is reserved for implementation work. Wait for required checks and merge it. The macOS build job and the tier lanes it feeds report `skipping` rather than `pass`, which is expected for a signed site artifact; the CI verdict check treats those skips as legitimate under the classifier's decision, and every other lane must still be terminal. The exact heading matters: placing `Refs #N` elsewhere in the body fails the repository validator.
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
- Interrupted matching draft publication: rerun the same `--publish` command; it safely resumes the draft.
- A GitHub HTTP 404 from `verify-remote-release-tag` means the reviewed tag has not been pushed yet; push that exact local tag, never recreate or move it.
- Conflicting remote release, asset, notes, digest, or tag: stop without clobbering anything.
- Failed Pages deployment or public-byte comparison: keep the prior feed active when possible, do not declare success, and repair through a pull request.

## Known gotchas (accumulated from release runs)

- `cargo xtask release accept-sparkle-update` rebuilds Sparkle's `sparkle-cli` with `xcodebuild` only when the prebuilt binary is missing. A refreshed checkout (wiped `build/` directory) therefore exposes a latent failure: Sparkle's Xcode project pins `MACOSX_DEPLOYMENT_TARGET=10.13`, which is below the minimum supported by current Xcode SDKs, and the build fails with exit 65. The command overrides the deployment target on the command line for this fixture-only build; release bytes are unaffected because the app and `generate_appcast` come from SwiftPM artifacts. If the acceptance preflight still fails at `resolve-sparkle-tools`, check for a new SDK minimum and bump the override in `resolve_sparkle_tools` (`xtask/src/release/sparkle_acceptance.rs`).
- The release-notes `mktemp` template carries no `X` placeholders: macOS `mktemp` never expands placeholders that sit before a suffix, so the old suffix-bearing template produced a literal `XXXXXX` filename. The template is now deterministic per tag, which means an aborted earlier run can leave the file behind; `mktemp` creates exclusively and fails on it, so remove the stale file and rerun.
- The PR issue compliance validator searches only the `## Linked issue` section once the heading exists: the section must carry at least one canonical reference (`Fixes|Closes|Resolves|Refs #N`; grouped forms like `Refs #N and #M` are accepted), and every `#N` inside it must use a canonical keyword — a bare `#N` in prose fails. Keep narrative above the heading. For release maintenance PRs (appcast activation, version bump) use `Refs #N`; `Fixes` is reserved for implementation work.
- Expected CI duration for integration verification PRs is roughly 5-8 minutes (live runs since #1348: the macOS build job just under 5 minutes, macOS Swift and Node contracts around 2-3 minutes (its packaging contracts step is a few-second Swift icon contract), the macOS build job also running every xtask unit test, and the per-tier test lanes another minute or two after the build finishes); do not treat a pending job at the 2-3 minute mark as stuck. That budget assumes a native-build cache hit: a GitHub runner-image rollout re-fingerprints the native identity (the toolchain probes are identity inputs), so the next PR can miss the native-build cache and pay a ~5-minute CMake rebuild, producing a roughly 11-minute job (hit live in the 0.2.70 release run). Before suspecting the release change, check the job log for `Cache not found for input keys: astronomical-v2-native-build-…` and compare the logged `[native-build-identity] … identity=…` value against the cached keys (`gh cache list --key astronomical-v2-native-build`); a changed identity with an unchanged `crates/native-identity/src` tree means GitHub rolled the image, not that the release broke something. Toolchain pinning to stop this drift is tracked separately (#1261).
- `cargo xtask release prepare-and-publish` takes the signing identity, team ID, and notary profile as explicit flags; `cargo xtask release prepare-signed` resolves them itself, which is why this procedure uses it.
- A version bump leaves `third-party/RUST_DEPENDENCY_NOTICES.digest`'s `input_sha256` memo behind because the root `Cargo.toml` is one of its recorded inputs; the notices text itself is unchanged by a version bump. The check passes on the text-match contract (log line `digest_memo=behind`), so committing `cargo xtask notices` output in the bump branch is optional fast-path hygiene, not a merge requirement (previously a hard CI failure, hit live in the 0.2.66 release run, fixed by #1395).
- The installed-variant round trip panics within a second with `workspace binary 'astronomical-inference-worker' was not found at …/target/release/…` when the release-profile binaries were never built: the test launches the real worker and daemon rather than building them. Run the `cargo build --release … --bins` line in section 3, step 2 first (hit live in the 0.2.89 release run). It also loads every installed chat model onto the graphics processor, so a model resident in the Development instance competes for the same memory; check `cargo xtask app instance status` and get the user's consent before stopping an instance that holds a model.
- `cargo xtask notices` refuses to run when the local `cargo-about` differs from the pin in `third-party/tool-pins/Cargo.toml` (`requires cargo-about 0.9.3; found cargo-about 0.9.2`). Install the pinned tools without touching the machine-wide copies: `cargo xtask install-verification-tools --prefix DIR`, then run the notices commands with `DIR` first on `PATH`.
- Any `Cargo.lock` change — including a dependency alignment that only removes a duplicate version — can leave `third-party/RUST_DEPENDENCY_NOTICES` stale; the nightly notices-freshness workflow then turns `main` red and blocks the "successful latest `main` CI" precondition (hit live in the 0.2.89 preparation: a tower-http alignment removed 0.7.1 from the lockfile without regenerating). Before the version bump, run `cargo xtask notices --check`; when it reports stale, it now lists the lines that moved (`+` only in the fresh harvest, `-` only in the committed file), so the cause is named rather than guessed. Regenerate with `cargo xtask notices` on a branch and merge that first.
- `cargo xtask release accept-sparkle-update` prints `Error: Update has failed due to error 4005 (SUSparkleErrorDomain). The update is improperly signed…` near the end of a passing run. That line is the acceptance's deliberate corrupted-update probe being rejected, as it must be; the run passed when it ends with `Sparkle acceptance passed: …` and `EXIT=0`.
- Before 0.2.89, `cargo xtask app package` (and so `prepare-signed`) had two latent failures that only a clean machine or the pinned toolchain exposed, both fixed in #1807 and now exercised weekly by the macOS packaging workflow (dispatch it before cutting a release: `gh workflow run macos-packaging.yml`): it only verified the pinned native archives and never downloaded them, so a fresh Mac must run `cargo xtask bootstrap-native-dependencies` first (the missing-archive error names that command); and it read the Thin Talk canvas resource bundle from `.build/out/Products/Release`, a layout only the swiftbuild backend creates, so a release built with the pinned Xcode's native SwiftPM failed at assembly. If a release build fails at a step the last weekly packaging run passed, compare the local Xcode and Swift versions with the pin before suspecting the release change.
- Removing a config field from the Rust structs is not enough: add the removed field name to `strip_retired_config_fields` in `crates/config/src/config_file.rs` and drop any dangling reference from `site/schemas/config/v1/astronomical-config.schema.json` in the same change, or upgraded installs fail strict parsing and the daemon exits before readiness (hit live in the 0.2.66 release run, fixed by #1116).

## FAQ

- Why did the Sparkle acceptance fail right after it worked in a previous release? The acceptance command skips `xcodebuild` when a prebuilt `sparkle-cli` exists. A prior release can succeed with the stale binary; the next checkout refresh deletes `build/`, forces the rebuild, and only then exposes the deployment-target incompatibility.
- Why do the macOS jobs show "skipping" on the appcast PR? The classify workflow marks signed site-artifact-only changes as not requiring the full integration run. The CI verdict check accepts those skips as legitimate under the classifier's decision; every other lane must still reach a terminal success state.
- Why does the `## Linked issue` section reject my body? Every `#N` inside the section must carry a canonical keyword (`Fixes|Closes|Resolves|Refs`); a bare `#N` buried in prose fails so release provenance stays auditable. Keyword-free narrative is accepted; keep it above the heading.
- Can I rerun a partially completed publish? Yes. Rerun the same `--publish` command with `--output-directory` pointing at the prepared directory; it resumes the matching draft publication safely.
- Why do verification steps run plain commands instead of a gate script? The commit-gate wrapper was retired (issue #983) in favor of the plain commands it wrapped: `cargo fmt --all -- --check` and `cargo test-integration-and-rest`. The Stable packaging contracts also stopped being explicit-only (issue #1346): every release step is now a typed `cargo xtask release` command whose decisions run as `cargo test -p xtask` unit tests on every pull request, alongside the icon renderer contract.
