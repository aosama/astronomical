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
- The app bundle holds two applications: the main app (`Contents/MacOS/astronomical-app`, the chat window, which owns Sparkle) and the menu-bar agent nested at `Contents/Library/LoginItems/Astronomical Agent.app` (which owns the engine). Both are signed, checked and notarized as one bundle by the commands below; never sign, staple or ship either on its own.
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

If any credential check fails, stop and read the message: the notarization check quotes notarytool's own reply. A reply about a missing or expired agreement means the account holder must accept an updated agreement on the Apple Developer account page, which shows it only as a banner there; Apple's services can then take a few minutes to recognise it, during which the check can pass and fail by turns. Any other failure: ask the user to unlock the Mac or repair the Keychain. Do not search chats for alternatives.

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

2. Run the release round trip only when the release changes how models load. It is the one gate that loads real installed models through the production worker and daemon: the smallest installed chat model and the smallest installed image model, not the whole inventory (the family-specific journeys cover each family's loader). It can only catch a model-loading regression and it costs stopping Development and a release build, so a release that touches no model-loading code skips it. Decide from the diff:

```sh
git diff --quiet "$PREVIOUS_TAG"..HEAD -- apps/inference-worker crates/model-serving crates/config crates/astronomical-jangtq2-qwen35 crates/quantization crates/vision-tower crates/neural-engine crates/mlx-c-fork crates/runtime-integration crates/astronomical-memory-policy crates/native-build-orchestration crates/native-build-tool crates/native-identity third-party/pins third-party/patches third-party/native-dependency-manifest.cmake registry; echo MODEL_LOADING_UNCHANGED=$?
```

`MODEL_LOADING_UNCHANGED=0` means none of those paths changed: skip the round trip and say so in the section 4 report. `1` means run it. A dependency update in `Cargo.lock` that moves MLX or a model-format crate also counts as a change. To run it, first read `cargo xtask app instance status`: if the Development instance holds a model, ask the user before stopping it with `cargo xtask app instance stop`, because the round trip loads two models onto the same graphics processor. The test launches the real worker and daemon binaries from `target/release`, so build them first:

```sh
cargo build --release --locked -p astronomical-inference-worker -p astronomical-daemon --bins
cargo test --release -p astronomical-acceptance --features serving-acceptance --test chat_serving should_round_trip_one_installed -- --ignored --nocapture --test-threads=1
```

The filter runs two journeys. `should_round_trip_one_installed_chat_model` loads the smallest chat model the Development daemon would serve (its own Library models folder plus the configured model directories) and requires a non-empty sampled reply (roughly one hundred tokens in at temperature 1.0, one-hundred-token budget). `should_round_trip_one_installed_image_model` loads the smallest image model and requires one 256-pixel PNG; with no image model installed it prints `status=skipped kind=image` and passes, which you then name in the section 4 report. Each has a 120-second budget, and each prints the model it chose. A model that fails to load, replies empty or returns no picture blocks the release; fix forward before tagging. These journeys need a real graphics processor and installed models, so they never run in continuous integration. The Development instance is the single source of the inventory; a model installed only in the Stable instance's Library is outside it. When the round trip passes, start Development again with `cargo xtask app instance start`. A stop also quits Development's chat app and its menu-bar agent; open the Development app bundle to bring both back (opening it shows the chat window and starts the agent), or leave it headless.

3. Run the cold-start window check when the release changes either app. Decide from the diff:

```sh
git diff --quiet "$PREVIOUS_TAG"..HEAD -- apps/astronomical-menu apps/thin-talk; echo APPS_UNCHANGED=$?
```

`APPS_UNCHANGED=0` skips it; say so in the section 4 report. `1` means run it, on the Development build only. Read `cargo xtask app instance status` first and ask before stopping an instance that holds a model, then:

```sh
cargo xtask app instance stop
cargo xtask app dev
```

`cargo xtask app dev` starts the chat app, the menu-bar agent and the engine in the same second, which is how every person's launch after an update goes. Take a screenshot of the Development chat window within a few seconds: the composer must show the chat model selector, the sidebar must list earlier conversations, and the window must show no failure text. Then open the app menu and confirm it lists "Check for Updates…". A window that fails this check blocks the release. Every automated test starts with an engine already running, so this is the only gate that sees a window that opened before its engine; 0.2.97 and 0.2.98 shipped with an empty model list for exactly that reason.

4. Require a clean worktree, workspace version `$VERSION`, and release notes that are nonempty and public-safe.
5. Build the Mac App Store bundle from this same commit, so a direct release never goes out while the store channel is broken:

```sh
cargo xtask app package --channel app-store --signing-identity "$DISTRIBUTION_IDENTITY"; echo EXIT=$?
```

`DISTRIBUTION_IDENTITY` is the Keychain identity whose quoted name starts with `Apple Distribution:` (section 8 lists the store credentials). The build uploads nothing and changes nothing in the worktree. It ends with the store bundle check: both applications sandboxed, the engine and worker inheriting, no Sparkle, no terminal command, the privacy manifest, the export declaration, and an engine that belongs to this commit. A nonzero exit stops the release here, before any tag exists; fix the cause through its own pull request and start this section again. If this Mac has no distribution identity, stop and tell the user: do not skip the step. Section 8 builds the bundle again from the tag for the upload itself; this run only proves the channel still builds.

6. Create a local lightweight tag with `git tag "$TAG"`. Do not push it until the publication approval gate passes.
7. Prepare the release:

```sh
cargo xtask release prepare-signed \
  --tag "$TAG" \
  --notes-file "$RELEASE_NOTES_FILE"
```

This authoritative command resolves the credential map, then signs inside-out (nested Sparkle code, the menu-bar agent's executable and its bundle, then the main bundle's executables), signs the app with Hardened Runtime and timestamps, creates and signs the DMG, submits it to Apple, staples and validates the ticket, validates the mounted installation layout, Sparkle-signs the final stapled bytes, signs the feed, and writes `target/releases/${TAG}/release-manifest.json`. Because it waits for Apple's `Accepted` verdict and validates the mounted DMG itself, section 4 does not repeat those checks. This output is long and carries the notarization submission ID that section 4 requires; if you background the run, keep its log so that ID and the final `Prepared signed release` line stay retrievable.

## 4. Mandatory publication approval gate

Validate `target/releases/${TAG}` without changing it:

- Manifest tag, version, and commit equal `$TAG`, `$VERSION`, and `HEAD`. The manifest's top-level keys are `tag`, `version`, `commit`, and `dmg`, `notes`, `appcast`, each of the last three an object with `name`, `sha256`, and `size`.
- DMG, notes, and appcast sizes and SHA-256 digests equal the manifest.
- The DMG name is `Astronomical-${VERSION}-macOS-arm64.dmg`.
- The notarization log for the submission ID from preparation reports zero issues: `xcrun notarytool log SUBMISSION_ID --keychain-profile "Astronomical Notarization"`. Preparation already required `Accepted`, so parse only the issue count (the log is JSON; count its `issues` array, which is null or empty when there are none); do not publish the complete log.
- Appcast XML is valid and contains `$VERSION`, a build number greater than the previous release, macOS `14.0` and above, `arm64`, the exact GitHub release URL, an enclosure Ed25519 signature, and a signed-feed envelope.
- The prepared directory contains exactly the DMG, copied Markdown notes, `appcast.xml`, and `release-manifest.json`.

Report only the version, commit, artifact names, sizes, digests, and validation outcomes, including the outcome of section 3's store bundle build. Ask the user to approve publication, unless they already approved this release in advance; either way, report these results. Do not push the tag or mutate GitHub release state until approved.

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
2. Require `cmp site/appcast.xml target/releases/${TAG}/appcast.xml` to succeed. Do not format or edit the XML. No tests rerun here: the only changes are this prepared file and the README link of step 3, the code it ships was tested in section 2, and its signing key was proven in section 3.
3. Point the `README.md` download button at this release's disk image: in the link around `site/assets/download-for-mac.svg`, replace the previous tag and version so the address reads `https://github.com/OWNER/REPO/releases/download/${TAG}/Astronomical-${VERSION}-macOS-arm64.dmg`. The DMG name carries the version, so no fixed address reaches the newest file and this one link is edited by hand each release. The Pages home page needs no edit: `site/assets/release-feed.js` reads the address from the appcast activated here. Both files are published content, so the pull request still skips the macOS lanes.
4. Inspect status, diff, and recent log; commit only `site/appcast.xml` and `README.md` with `release: activate v${VERSION} update feed`.
5. Push and open a PR whose `## Linked issue` section contains exactly `Refs #${ISSUE_NUMBER}` (see the gotchas for the validator's rules). Wait for required checks and merge it. The macOS build job and the tier lanes it feeds report `skipping`, which the CI verdict accepts for a signed site artifact; every other lane must still be terminal.
6. Synchronize local `main` with the squash-merged remote commit without rebasing. A clean local branch can diverge by one commit after the remote squash merge; fetch `origin/main`, then use a fast-forward or clean reset to the remote branch. Wait for the Pages workflow for the merge commit, and require success.
7. Download the public `appcast.xml` to a temporary file. Require it to be byte-identical to `target/releases/${TAG}/appcast.xml` and to match the manifest digest. Verify its enclosure URL answers with a success HTTP status; section 5 already proved the public DMG's digest. Delete temporary files.

## 7. Finish

Require all of the following before reporting success:

- GitHub release `$TAG` is public and is the latest Stable release.
- Public Pages appcast is the exact prepared signed appcast.
- Every public "Download for Mac" button passes the download button checklist below.
- The `Announce release on Discord` workflow run for this release succeeded. Publishing the release starts it; it posts the notes to the Discord server's releases channel under the project's name and logo, so no release link is pasted there by hand. If it failed, run it again from the Actions tab with the tag as input.
- The Discord invite linked from the README and the website still works. Read the invite code from the README's Discord link and look it up: `curl -s "https://discord.com/api/v10/invites/CODE?with_expiration=true"` must name the server and show `"expires_at": null`. An invite made with Discord's defaults expires after 30 days, and an expired one fails silently for every newcomer; if it has expired, the server owner creates one set to never expire and the links are updated through a pull request.
- Release issue is explicitly closed after the appcast PR merge, Pages deployment, and public-byte verification. A `Refs #N` maintenance PR does not close it automatically.
- Local `main` equals remote `main` and the tracked worktree is clean.
- The ignored `target/releases/${TAG}` evidence directory remains intact.
- The real Astronomical release was not installed or launched locally; isolated temporary validation fixtures are allowed as described above.

### Download button checklist

A person who clicks "Download for Mac" must receive this release's disk image, not a page that lists it. Check the public copies, after the Pages deployment, never the working tree:

- [ ] **`README.md` on `main`:** the link around `site/assets/download-for-mac.svg` points at this release's DMG.
- [ ] **The Pages home page (`site/index.html`, served at the repository's GitHub Pages address):** both "Download for Mac" buttons (the hero and the closing call to action) point at this release's DMG. `site/assets/release-feed.js` sets these links from the appcast after the page loads, so read them from the page as rendered in a browser; the page source alone shows only the releases page, which is the address for a visitor whose feed did not load.
- [ ] **A click downloads the file:** each of those links, followed through its redirects, ends in the DMG itself.

Read the README link from GitHub and the two button links from the browser (`document.querySelectorAll('a[data-latest-disk-image]')`), then probe each without downloading the body:

```sh
gh api "repos/{owner}/{repo}/readme" -H "Accept: application/vnd.github.raw" | grep -o '<a href="[^"]*"><img src="site/assets/download-for-mac.svg"'
curl -sIL -o /dev/null -w '%{http_code} %{content_type} %header{content-length} %header{content-disposition}\n' "$BUTTON_URL"
```

A link passes only when the probe prints status `200`, type `application/octet-stream`, the manifest's DMG size, and a `content-disposition` naming `Astronomical-${VERSION}-macOS-arm64.dmg`. A `text/html` answer means the link opens a web page, such as `releases/latest`, and fails the check even though the page lists the DMG. A link naming an older version also fails. The DMG name carries the version, so the passing address is `https://github.com/OWNER/REPO/releases/download/${TAG}/Astronomical-${VERSION}-macOS-arm64.dmg`.

Section 6 step 3 is where the README link is set. Repair a failing link through its own pull request with `Refs #${ISSUE_NUMBER}`, wait for the Pages deployment when `site/` changed, and probe again before closing the release issue.

Tell the user that Astronomical `$VERSION` is available through the normal public release and Sparkle channels. Do not acquire it for them; they will receive it through the ordinary user journey.

## 8. Ship the same version to the Mac App Store

The store version is the same product built as a sandboxed channel. Do this stage after section 7 succeeds, from the release tag, so both channels carry one version number. It never changes the direct release.

Read the `mac-app-store` skill (`.claude/skills/mac-app-store/SKILL.md`) before the first step. It holds the detail behind each step here: the decisions already made for the app record, how to run and try the sandboxed build on screen, what Apple's validation has said before, and how to work on the App Store Connect pages.

Three steps need the user's word each time, and nothing else in this stage does: uploading a build, submitting for review, and releasing to the public. Ask in the chat; approval for a direct release does not cover them.

Store credentials, besides the map above:

- **Distribution identity:** the Keychain identity whose quoted name starts with `Apple Distribution:`.
- **Installer identity:** the one whose quoted name starts with `3rd Party Mac Developer Installer:`.
- **App Store Connect API key:** the key's private half lives in `~/.appstoreconnect/private_keys/AuthKey_<KEY ID>.p8`. The Key ID is in that file name; the Issuer ID is on the Users and Access, Integrations page of App Store Connect. Never copy the private key anywhere else or print it.

Steps:

1. Build the store bundle from the clean release tag: `cargo xtask app package --channel app-store --signing-identity "$DISTRIBUTION_IDENTITY"`. The bundle check refuses a bundle with an unsandboxed executable, a terminal command, a missing privacy manifest, missing entitlements, or a missing export declaration.
2. Prove it runs sandboxed. Stop the installed Stable app first (check it is idle; the two share a port), copy the bundle into the Applications folder under another name, and run `cargo xtask app smoke-test --app-bundle "/Applications/<that name>.app"`. From a build folder a sandboxed agent cannot reach its engine, so the smoke test must run on the Applications copy. Then try each promise of the listing on screen, following the table in the `mac-app-store` skill; the smoke test does not open a file dialog. Move the copy to the Trash and reopen Stable afterwards.
3. Package it: `cargo xtask release app-store-pkg --app-bundle PATH --installer-identity "$INSTALLER_IDENTITY"`.
4. Ask Apple to validate it: `cargo xtask release app-store-upload --package PATH --api-key KEY_ID --api-issuer ISSUER_ID`. Validation sends nothing permanent. Fix every error it reports before going on. One warning is expected until provisioning profiles exist: 90889, which only concerns TestFlight.
5. **Ask the user, then upload:** the same command with `--upload`. An uploaded build cannot be withdrawn, and a build number can be used once per version. Processing takes roughly ten to thirty minutes before the build can be selected.
6. On the app record in App Store Connect, create the new version with the release's version number, write "What's New" from the release notes in plain words, and select the uploaded build. Screenshots carry over from the previous version; replace them only when the app looks different.
7. **Ask the user, then submit for review.** Keep release set to manual.
8. When Apple answers: on approval, install the approved build from the store on a Mac account that has never run the store version and reach a first reply before asking the user whether to release; on rejection, record Apple's reasons on the release issue and open one issue per reason.

Rules for the app record that must not change by accident:

- **Availability excludes the European Union.** The account holder has declared to Apple that they do not distribute there. When availability is edited, all 27 member states stay unticked, automatic availability in future countries stays off, and "distribution in France" is answered "No".
- **Listing text and screenshots** promise only what works in the sandboxed build. Before adding a promise, do it on screen in the store build.
- **The privacy answer is "Data Not Collected".** It must change in the same release as any feature that sends data anywhere.
- **Entitlements are part of the review.** A new entitlement needs a sentence in the review notes saying why.

To replace a build that is already waiting for review, remove the version from review on the app record first, then repeat steps 1 to 7 with the new build.

## Failure rules

- Before tag push: fix the cause, remove only an invalid `target/releases/${TAG}` after explicit user approval, recreate the local tag only if its commit is wrong, and prepare again.
- After tag push: never move the tag; withhold appcast activation and fix forward with a new patch release.
- Interrupted matching draft publication: rerun the same `--publish` command; it safely resumes the draft.
- A GitHub HTTP 404 from `verify-remote-release-tag` means the reviewed tag has not been pushed yet; push that exact local tag, never recreate or move it.
- Conflicting remote release, asset, notes, digest, or tag: stop without clobbering anything.
- Failed Pages deployment or public-byte comparison: keep the prior feed active when possible, do not declare success, and repair through a pull request.

## Known gotchas

- The Sparkle acceptance in section 3 uses a single-process fixture app, so it proves the signing key and the update mechanics but not the hand-off between the main app and its agent. During a real update the main app tells the agent to quit (which stops the engine) before Sparkle relaunches it, and the relaunched app starts the new agent; an agent left running from an older build quits itself when the new main app sends its build check. When a release changes that hand-off (`stopAgentBeforeUpdate` in `apps/astronomical-menu/Sources/AstronomicalAppCore/AstronomicalApplication.swift`, `ProcessCommand` in `Sources/AstronomicalShared`, or `SparkleApplicationUpdateController`), say so in the section 4 report as not covered by any automated gate, and exercise it on a Development build first: `cargo xtask app dev`, then check the app menu's Quit and Stop Engine leaves no `astronomical-menu` or `astronomicald` process.
- `validate-distribution-app` (run inside `prepare-signed`) requires the agent's executable and bundle by their Stable names, and the bundle check refuses a main app marked `LSUIElement`, an agent that is not, or an agent whose channel, port, version or commit differ from the main app's. A failure naming the agent means the Swift build produced only one of the two products; rebuild rather than editing the bundle.
- Launch at Login registers the agent, not the app. The Development channel hides the control, so it cannot be tried on a Development build; the first launch of a release moves a registration made by a release before the agent existed onto the agent.

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
