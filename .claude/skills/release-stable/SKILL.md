---
name: release-stable
description: Prepare, sign, notarize, publish and activate one Astronomical Stable release. Use when asked to make, cut or ship a release or a new version of Astronomical, and read it before tagging or running any `cargo xtask release` step. Not for ordinary builds, commits or pull requests.
---

# Release Astronomical Stable

Release the version the user named. If they named none, ask for a numeric `MAJOR.MINOR.PATCH`; never infer it. Then use the confirmed value as `VERSION` and `TAG="v${VERSION}"`, and require it to be newer than the latest semantic release tag. Asking for a release authorizes the release work, but you must stop once after local artifact review and obtain the user's explicit approval before pushing the tag or publishing. Approval the user gave in advance for this release counts; the review itself still happens, and its results still go in your report.

## Non-negotiable rules

- Work from the repository root and follow `AGENTS.md`.
- Never use chat history, chat-memex, memory, or guesswork for credential names.
- Never commit or place credential values, local paths, Keychain output, or machine logs in GitHub content or the final report. Do not suppress authoritative release-command output merely because it contains local operator paths.
- Use only `cargo xtask release prepare-signed` to prepare or publish distribution artifacts. It is the credential-resolving front door of `cargo xtask release prepare-and-publish`, which takes the signing identity, team ID, and notary profile as explicit flags instead.
- Never manually edit `site/appcast.xml`; it is a signed generated artifact.
- Never move or replace a pushed release tag. Fix forward with a new patch version.
- Read this whole procedure before the first step. Do not push the tag before section 4 passes, even when approval was given in advance: a pushed tag can never move, so anything section 3 or 4 finds afterwards costs a patch version.
- Do not promote the real release into `~/Applications` or launch it. Temporary mounting and copying by the DMG validation inside `cargo xtask release prepare-signed`, and installation and launch of the fictional fixture under `target/` by `cargo xtask release accept-sparkle-update`, are required isolated validation—not installation of the release. The developer must later receive the real app through the same public download or Sparkle update journey as every user.
- Every test must use its built-in timeout of at most 120 seconds. The Sparkle acceptance's 120-second budget covers the acceptance itself; resolving its Swift and Xcode tools beforehand reports its own progress. The notarization operation is not a test and uses `cargo xtask release notarize-dmg`'s 180-second default, overridable with `--timeout-seconds N`.
- Keep commands live and unfiltered. Never pipe long-running commands through `head`, `tail`, or another buffering filter.

## Credential map

Use these sources exactly; no other credential discovery is allowed:

- **Developer ID identity:** the single Keychain code-signing identity whose quoted name starts with `Developer ID Application:`.
- **Team ID:** the 10-character value in the final parentheses of that identity.
- **Notarization profile:** `Astronomical Notarization`.
- **Sparkle private key:** the Keychain key automatically selected by Sparkle's `generate_appcast`. `cargo xtask release accept-sparkle-update` proves that it matches Astronomical's embedded public key.

`cargo xtask release prepare-signed` resolves this map itself, failing closed unless exactly one Developer ID Application identity and a 10-character Team ID are found, and it checks the notarization profile is readable before any build starts, so no separate profile preflight is needed. Use it instead of resolving the values by hand.

If any credential check fails, stop and ask the user to unlock the Mac or repair the Keychain. Do not search chats for alternatives.

## 1. Establish scope and issue

1. Fetch remote state; require clean synchronized `main`, successful latest `main` CI, no existing `$TAG`, and no existing GitHub release for `$TAG`.
2. Set `PREVIOUS_TAG` to the highest existing semantic release tag and review `git log` and `git diff` from it to `main`.
3. Create one public issue titled `Release Astronomical ${VERSION}`. State all changes, enhancements, bug fixes, compatibility, signing requirements, digest-bound publication, signed-feed activation, and public verification. Do not include private environment details. Also in the release notes acknowledge contributors who opened GitHub issues that this release addressed.
4. Save the returned issue number as `ISSUE_NUMBER`. Both release pull requests must link only this issue.
5. Compare `site/schemas/config/v1/astronomical-config.schema.json` at `PREVIOUS_TAG` with `main`, and require every removed property to be in `strip_retired_config_fields` (`RETIRED_CHUNKING_FIELD_NAMES` for chunking fields).
6. Merge any fix the release needs on its own branch and pull request before starting section 2, so the bump stays version-only. A fix mixed into the bump branch fails the version-only classifier, and the bump then pays for the full macOS lanes.

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
7. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}`. Wait for required checks (the change classifier proves from the file contents that this diff is only a version bump, so the macOS build and test lanes report `skipping` and the PR finishes in a few minutes; see the gotchas), merge it, then synchronize the release commit: in a primary checkout, switch to `main` and pull with `--ff-only`; in a linked worktree whose primary checkout owns `main`, fetch and check out the merge commit directly (`git checkout --detach origin/main`). Either way, verify local `HEAD`, `refs/remotes/origin/main`, and GitHub `main` resolve to the same commit. Hold other pull requests (and their auto-merge) from the bump's merge until the tag is pushed: anything that lands in between becomes part of the release commit without having been in the bump's local suite.

## 3. Preflight and prepare locally

1. Prove the Sparkle signing key matches the app's embedded public key. This is the only run of the acceptance in a release, because section 6 activates the appcast this key signed without changing it:

```sh
cargo xtask release accept-sparkle-update; echo EXIT=$?
```

Its transcript is long, so the explicit exit code keeps the result in reach. Near the end of a passing run it prints `Error: Update has failed due to error 4005 (SUSparkleErrorDomain)…`: that is the deliberate corrupted-update probe being rejected. The run passed when it ends with `Sparkle acceptance passed: …` and `EXIT=0`.

2. Run the installed-variant round trip — the gate that catches pack variants the single-pack journeys structurally miss (a real installed export whose config shape the loader refuses must fail here, before the release ships). First read `cargo xtask app instance status`: if the Development instance holds a model, ask the user before stopping it with `cargo xtask app instance stop`, because the round trip loads every installed model onto the same graphics processor. The test launches the real worker and daemon binaries from `target/release`, so build them first:

```sh
cargo build --release --locked -p astronomical-inference-worker -p astronomical-daemon --bins
cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving should_round_trip_every_installed_chat_model_variant -- --ignored --nocapture --test-threads=1
```

Every chat-capable model the Development daemon would serve (its own Library models folder plus the configured model directories) must load and stream a non-empty sampled reply (roughly one hundred tokens in at temperature 1.0, one-hundred-token budget, 120-second budget per model). The Development instance is the single source of the variant inventory. A variant that fails to load or replies empty blocks the release; fix forward before tagging. This journey needs a real graphics processor and installed model inventory — it never runs in continuous integration. A model installed only in the Stable instance's Library is outside this inventory and is not exercised; when the developer uses such a model, name it as not covered in the section 4 report. When the round trip passes, start Development again with `cargo xtask app instance start` and reopen its menu app bundle, which a stop closes.

3. Require a clean worktree, workspace version `$VERSION`, and release notes that are nonempty and public-safe.
4. Create a local lightweight tag with `git tag "$TAG"`. Do not push it until the publication approval gate passes.
5. Prepare the release:

```sh
cargo xtask release prepare-signed \
  --tag "$TAG" \
  --notes-file "$RELEASE_NOTES_FILE"
```

This authoritative command resolves the credential map, then signs nested Sparkle code and Astronomical executables inside-out, signs the app with Hardened Runtime and timestamps, creates and signs the DMG, submits it to Apple, staples and validates the ticket, validates the mounted installation layout, Sparkle-signs the final stapled bytes, signs the feed, and writes `target/releases/${TAG}/release-manifest.json`. Because it waits for Apple's `Accepted` verdict and validates the mounted DMG itself, section 4 does not repeat those checks. This output is long and carries the notarization submission ID that section 4 requires; if you background the run, keep its log so that ID and the final `Prepared signed release` line stay retrievable.

## 4. Mandatory publication approval gate

Validate `target/releases/${TAG}` without changing it:

- Manifest tag, version, and commit equal `$TAG`, `$VERSION`, and `HEAD`. The manifest's top-level keys are `tag`, `version`, `commit`, and `dmg`, `notes`, `appcast`, each of the last three an object with `name`, `sha256`, and `size`.
- DMG, notes, and appcast sizes and SHA-256 digests equal the manifest.
- The DMG name is `Astronomical-${VERSION}-macOS-arm64.dmg`.
- The notarization log for the submission ID from preparation reports zero issues: `xcrun notarytool log SUBMISSION_ID --keychain-profile "Astronomical Notarization"`. Preparation already required `Accepted`, so parse only the issue count (the log is JSON; count its `issues` array, which is null or empty when there are none); do not publish the complete log.
- Appcast XML is valid and contains `$VERSION`, a build number greater than the previous release, macOS `14.0` and above, `arm64`, the exact GitHub release URL, an enclosure Ed25519 signature, and a signed-feed envelope.
- The prepared directory contains exactly the DMG, copied Markdown notes, `appcast.xml`, and `release-manifest.json`.

Report only the version, commit, artifact names, sizes, digests, and validation outcomes. Ask the user to approve publication, unless they already approved this release in advance; either way, report these results. Do not push the tag or mutate GitHub release state until approved.

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
2. Require `cmp site/appcast.xml target/releases/${TAG}/appcast.xml` to succeed. Do not format or edit the XML. No tests rerun here: the only change is this prepared file, the code it ships was tested in section 2, and its signing key was proven in section 3.
3. Inspect status, diff, and recent log; commit only `site/appcast.xml` with `release: activate v${VERSION} update feed`.
4. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}` (see the gotchas for the validator's rules). Wait for required checks and merge it. The macOS build job and the tier lanes it feeds report `skipping`, which the CI verdict accepts for a signed site artifact; every other lane must still be terminal.
5. Synchronize local `main` with the squash-merged remote commit without rebasing. A clean local branch can diverge by one commit after the remote squash merge; fetch `origin/main`, then use a fast-forward or clean reset to the remote branch. Wait for the Pages workflow for the merge commit, and require success.
6. Download the public `appcast.xml` to a temporary file. Require it to be byte-identical to `target/releases/${TAG}/appcast.xml` and to match the manifest digest. Verify its enclosure URL answers with a success HTTP status; section 5 already proved the public DMG's digest. Delete temporary files.

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

## Known gotchas

- The PR issue compliance validator searches only the `## Linked issue` section once the heading exists: the section must carry at least one canonical reference (`Fixes|Closes|Resolves|Refs #N`; grouped forms like `Refs #N and #M` are accepted), and every `#N` inside it must use a canonical keyword, so a bare `#N` in prose fails. Keep narrative above the heading. Release maintenance PRs (version bump, appcast activation) use `Refs #N`; `Fixes` is reserved for implementation work.
- The version bump PR runs only the static, configuration-contract, and notices lanes. The change classifier (`xtask/src/version_only_bump.rs`) recognises a bump only when `Cargo.toml` differs solely in the `[workspace.package]` version, `Cargo.lock` solely in workspace packages' versions, and no file changes beyond those two and the notices digest; anything else, or any read failure, runs the full matrix. The macOS lanes therefore show `skipping` on the bump PR and on its merge to `main`, and the CI verdict accepts that. Section 2 step 5's local suite is the only full run on the bumped commit, so never skip it: a test that hard-codes the version string would pass CI and fail there.
- The section 2 suite and the section 3 release build take the longest local wall time. Running them once on a tree identical to the coming release commit, while the last fix pull request's CI runs, warms the build caches so the required runs on the bumped commit finish in a fraction of the time; it never replaces those required runs.
- A merge to `main` also shows the macOS lanes as `skipping` when its exact tree already passed them on its pull request and no cache input changed since `main` last ran them within five days (`xtask/src/verified_tree.rs`). Its CI verdict still counts as the successful latest `main` CI.
- Integration verification PRs take roughly 5-8 minutes with a native-build cache hit; do not treat a pending job at the 2-3 minute mark as stuck. A GitHub runner-image rollout re-fingerprints the native identity, so the next PR can miss the cache and take about 11 minutes. Before suspecting the release change, look for `Cache not found for input keys: astronomical-v2-native-build-…` in the job log and compare the logged `[native-build-identity] … identity=…` with `gh cache list --key astronomical-v2-native-build`; a changed identity with an unchanged `crates/native-identity/src` tree means GitHub rolled the image (toolchain pinning is tracked in #1261).
- Any `Cargo.lock` change, even one that only removes a duplicate version, can leave `third-party/RUST_DEPENDENCY_NOTICES` stale; the nightly notices-freshness workflow then turns `main` red and blocks section 1's CI precondition. Before the version bump, run `cargo xtask notices --check`; it lists the lines that moved. Regenerate with `cargo xtask notices` on a branch and merge that first.
- `cargo xtask notices` refuses to run when the local `cargo-about` differs from the pin in `third-party/tool-pins/Cargo.toml`. Install the pinned tools without touching the machine-wide copies with `cargo xtask install-verification-tools --prefix DIR`, then run the notices commands with `DIR` first on `PATH`.
- The release-notes path in section 2 step 4 is fixed per tag, so an aborted earlier run leaves the file behind and `mktemp` refuses to overwrite it; remove the stale file and rerun.
- If `cargo xtask release accept-sparkle-update` fails at `resolve-sparkle-tools` after a checkout refresh, Xcode's SDK minimum has likely passed the deployment-target override used for the fixture-only `sparkle-cli` build; bump it in `resolve_sparkle_tools` (`xtask/src/release/sparkle_acceptance.rs`). Release bytes are unaffected.
- Do not wait on the macOS packaging workflow before a release: `prepare-signed` packages the release for real and fails before any tag is pushed, and the workflow runs weekly on its own. A fresh Mac must run `cargo xtask bootstrap-native-dependencies` before packaging (the missing-archive error names that command). If a release build fails at a step the last weekly packaging run passed, compare the local Xcode and Swift versions with the pin before suspecting the release change.
- Section 1 step 5's schema comparison exists because a config field removed from the Rust structs but missing from `strip_retired_config_fields` (`crates/config/src/config_file.rs`) makes upgraded installs fail strict parsing, and the daemon exits before readiness. A removal adds the field there and drops any dangling reference from `site/schemas/config/v1/astronomical-config.schema.json` in the same change.
