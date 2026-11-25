---
name: macos_appstore_release
description: Build, test, validate, upload and submit the Mac App Store version of Astronomical, and look after its App Store Connect record. Use when asked about the store build, the sandbox, App Store Connect, App Review, a rejection, screenshots or listing text, or when a release has to go to the store. Read all of it before building or touching the record. Only when the user asks for a store release in their own words. Not for the direct download and Sparkle update, which is the sparkle_release skill.
---

# Astronomical in the Mac App Store

The store version is the same product built as a sandboxed channel, listed as **Astronomical: Local AI**. The direct download stays the primary way to get the app; the store is a second door for people who trust it more.

This skill holds what was learned getting the first version into review, so that nobody has to find it out again. "Steps of a store release" below lists them in order; the rest of this file is the detail behind each of them. The issue map is aosama/astronomical#381.

Two files sit beside this one:

- [listing.md](listing.md): the listing text as it stands on the record.
- [review-notes.md](review-notes.md): the notes App Review reads, and what must change in them when the app changes.

## Steps of a store release

The Astronomical skill (`skill/astronomical`) is reviewed in the Sparkle release of the same version (section 1 of the `sparkle_release` skill). A store release of a version that never had a Sparkle release does that review first.

Start only when the user asks for a store release in their own words; a Sparkle release never implies one. Build from the tag of a version the `sparkle_release` skill has already published, so both channels carry one version number. A store release never changes the direct release.

The sections below this one hold the detail behind each step: the decisions already made for the app record, how to run and try the sandboxed build on screen, what Apple's validation has said before, and how to work on the App Store Connect pages.

Three steps need the user's word each time, and nothing else in this procedure does: uploading a build, submitting for review, and releasing to the public. Ask in the chat; approval for a Sparkle release, even approval given in advance, does not cover them.

Store credentials, besides the Developer ID map in the `sparkle_release` skill:

- **Distribution identity:** the Keychain identity whose quoted name starts with `Apple Distribution:`.
- **Installer identity:** the one whose quoted name starts with `3rd Party Mac Developer Installer:`.
- **App Store Connect API key:** the key's private half lives in `~/.appstoreconnect/private_keys/AuthKey_<KEY ID>.p8`. The Key ID is in that file name; the Issuer ID is on the Users and Access, Integrations page of App Store Connect. Never copy the private key anywhere else or print it.

Steps:

1. Build the store bundle from the clean release tag: `cargo xtask app package --channel app-store --signing-identity "$DISTRIBUTION_IDENTITY"`. The bundle check refuses a bundle with an unsandboxed executable, a terminal command, a missing privacy manifest, missing entitlements, or a missing export declaration.
2. Prove it runs sandboxed. Stop the installed Stable app first (check it is idle; the two share a port), copy the bundle into the Applications folder under another name, and run `cargo xtask app smoke-test --app-bundle "/Applications/<that name>.app"`. From a build folder a sandboxed agent cannot reach its engine, so the smoke test must run on the Applications copy. Then try each promise of the listing on screen, following the table under "What to try on screen before any submission" below; the smoke test does not open a file dialog. Move the copy to the Trash and reopen Stable afterwards.
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

## What needs the account holder, every time

Ask in the chat and wait for a clear yes before each of these. A yes for one does not cover another, and a yes for a direct release covers none of them.

| Step | Why it is theirs |
| --- | --- |
| Uploading a build | An uploaded build cannot be withdrawn, and a build number is spent |
| Submitting or resubmitting for review | It is a statement to Apple in their name |
| Removing a version from review | It gives up the place in the queue |
| Releasing to the public | Release is set to manual on purpose |
| Any legal declaration | Export, content rights, age rating, privacy, trader status, agreements |
| Credentials | Signing in, API keys, two-factor codes |
| Money | The record is free; any price or paid agreement is theirs |

Everything else is yours to do without asking: building, testing, validating, filling in text, uploading screenshots, writing issues.

Before choosing anything on the record, read what the account already says. The account holder has signed agreements and made declarations, and a default in a form can contradict them.

## Decisions already made

Do not ask again. Change one only when the account holder says so.

| Subject | Decision |
| --- | --- |
| Name | Astronomical: Local AI (the plain name was taken) |
| Subtitle | Run AI models on your Mac |
| Price | Free |
| Availability | 148 countries and regions. **The 27 European Union member states are excluded**, and availability in future countries is off |
| Trader status | Declared to Apple as not a trader and not distributing in the European Union |
| Categories | Productivity first, Developer Tools second |
| Copyright | 2026 Astronomical contributors |
| Release | Manual, after approval |
| Privacy label | Data Not Collected |
| Privacy policy | The privacy page of the project website |
| Age rating | 13+, from honest answers for an app whose models are not filtered: mild or infrequent for the content themes, and yes to health or wellness topics |
| Export | Standard encryption only; not distributed in France. The store build declares this itself (see below) |
| Content rights | Contains third-party content, with the rights to use it. Every catalog model is Apache-2.0 or MIT |
| Screenshots | Three, 2880 by 1800, each with a caption. The generated rocket poster comes first because it catches the eye |
| Listing tone | Inviting, plain, no jargon. It promises only what works in the sandboxed build |
| Model folders outside the app | Not offered in the store version (aosama/astronomical#2187, not planned). Its customer gets models inside the app and does not manage model files by hand; someone who does is served by the direct download |

### The European Union rule

The account holder declared to Apple that they do not distribute in the European Union. Whenever availability is edited: all 27 member states stay unticked, "make available in future countries" stays off, and the encryption question about France is answered "No". Choosing "all countries" would break their declaration.

### The export declaration

The store build's main `Info.plist` carries `ITSAppUsesNonExemptEncryption` set to false, so App Store Connect does not hold each upload for the encryption questions. It is written by `xtask/src/app_bundle/info_plist.rs` and required by the bundle check. It restates the account holder's answer. If the app starts using encryption beyond HTTPS and hashing, or France is added to availability, the answer is theirs to give again before the key changes.

### Who the store version is for

Someone who is not technical and wants no friction: they install, press Get started, and chat. Before adding anything to the store version, name the store customer who would use it. A feature that assumes the terminal, other model tools, or files arranged by hand belongs to the direct download.

## How the store build differs

| | Direct download | Store build |
| --- | --- | --- |
| Channel | `stable` | `app-store` |
| Sandbox | Off | On for both applications; the engine and worker inherit it |
| Updates | Sparkle | The store; no Sparkle in the bundle |
| Terminal command | Bundled | Left out, because it cannot run sandboxed |
| Where settings, logs and models live | `~/.astronomical` | The app group container: `~/Library/Group Containers/<team>.astronomical/Astronomical` |
| Where chat sessions live | Same folder | The chat app's own container: `~/Library/Containers/app.astronomical.appstore/Data/.astronomical` |
| Folder settings | Shown | Hidden; the sandbox cannot honour a folder typed into a field |

Things that are the same and matter: the engine reports the `stable` instance, and both versions use port 6732 and the `astronomical://` link scheme. So the two cannot run at the same time.

Bundle identifiers: `app.astronomical.appstore` for the chat app and `app.astronomical.appstore.agent` for the menu-bar helper. The team identifier is part of the app group name; never write it in the repository or in GitHub content.

Entitlements live in `scripts/internal/entitlements/`:

- `app-store-application.entitlements`, for the two applications: App Sandbox, outgoing connections, incoming connections, and files the person selects (read and write). The app group is added while signing (`xtask/src/app_bundle/application_group.rs`), because its name carries the team identifier.
- `app-store-inherited.entitlements`, for the engine and worker: App Sandbox and inherit, exactly those two.

## Build, run and test

All commands run from the repository root. `.cargo/config.toml` documents each under the `xtask` alias.

1. **Build** from a clean tree (the build number is the commit count, so uncommitted work is refused):
   `cargo xtask app package --channel app-store --signing-identity "$DISTRIBUTION_IDENTITY"`
   The identity is the Keychain one whose name starts with `Apple Distribution:`.
2. **Stop the installed direct app** if it is running, because the two share a port. Read its status first; if it is busy, ask. Quit the app and its helper, and reopen the app when you are done.
3. **Copy the bundle into the Applications folder** under another name, for example `Astronomical Store Test.app`, with `ditto`. A sandboxed helper started from a build folder cannot reach its engine, and the window waits on "Starting up" until it gives up. Only a copy in Applications behaves like an install.
4. **Smoke test:** `cargo xtask app smoke-test --app-bundle "/Applications/<that name>.app"`.
5. **Use it on screen** (next section).
6. **Clean up:** move the test copy to the Trash, reopen the direct app.

### A fresh install

Move these two folders to the Trash with the store app stopped: the `Astronomical` folder inside the app group container, and `.astronomical` inside the chat app's container `Data` folder. The next launch is a first launch. Downloaded models go with the first folder, so a fresh run downloads again.

### What to try on screen before any submission

Each of these is a promise in the listing. Tests in code do not cover the sandbox; a build once went to review unable to show any Open or Save dialog while every test passed.

| Promise | What to do | It works when |
| --- | --- | --- |
| First run | Fresh install, Get started, Download | A first reply arrives without reading any documentation |
| A different first model | Fresh install, "Choose a different model", get one on the Models page, come back | The chat window notices the model and moves on |
| Chat | Ask a question | A reply streams |
| Attach a photo | Ask about a photo | The picker opens, the reply describes the photo |
| Attach a file | Attach a text file | The reply uses its content |
| Make a picture | Ask for a picture | The picture model is offered, downloads, and the picture appears |
| Save a picture | Save it to a folder | The file is on disk |
| Export a conversation | Export as PDF | The file is on disk |
| Speech | Press Speak on a reply | It is read aloud |
| Models page | Get a model, remove a model | The list and free space update |
| Local API | Connect page, then a request to the address it shows | Only chat models are listed; the request answers |
| Settings | Open Settings | No Locations section |
| Launch at Login | Turn it on and off | The system setting follows |
| Two installs | Start it while another install's engine holds the address | The window and the menu-bar item say another copy is running, and carry on once it quits |
| Model link | `open -g -a "/Applications/<test copy>.app" "astronomical://models/huggingface/OWNER/NAME"` | The Models page opens in the browser on that model |
| Neural Engine switch | Tuning, Experimental, turn it on, Save, send a prompt of several thousand words | The reply arrives at once on the graphics chip; the status shows programs being prepared, then "active" |

Wait out a screen's own timers before calling it stuck. "Starting up" gives up by itself after fifteen seconds.

Notes on three of the rows:

- **Model link.** With the direct app also installed, macOS hands every `astronomical://` link to the direct app, so name the test copy with `-a` as shown. Without `-a` the link starts the direct app.
- **Neural Engine switch.** On a 9B model the 56 programs took about two and a half minutes to prepare in the sandbox. They cost about 10 GB in the app group container (`neural-engine-programs`) and the same again in the `coreml_model_cache` inside the helper's container (`Data/Library/Caches/…/com.apple.e5rt.e5bundlecache`). Turn the switch off again and move both folders to the Trash afterwards.
- **Two installs.** See the next paragraph.

The "Two installs" row is tried without the installed direct app: a small local server that answers `/health`, `/v1/status` and `/v1/models` on the engine's port, and reports the right channel with another state folder, plays the other install. Stop it and the app must recover without a relaunch.

### Limits of on-screen control

- Open test windows in the background (`open -g`, `cargo xtask app dev --background`). A window opened in front lands on top of whatever the person at the Mac is doing.
- A window captured in the first seconds after a background launch can come back blank white. That is the capture, not the app: in front, the window paints dark at once. Capture again before filing anything.
- Background app control cannot see or press buttons in a sandboxed app's Open and Save dialogs. Full-screen control can click Save, but not after a file name has been typed into the dialog; accept the suggested name.
- A screenshot of one window, when the tools cannot see it: `screencapture -x -o -l <window id>`.
- Screenshots fail while the display is asleep.
- In zsh, `log` is a builtin; call `/usr/bin/log` to read the system log, which names a missing entitlement plainly.
- The store engine cannot be started by hand, not even for `--version`: outside its app macOS kills it during sandbox setup (exit code 133, and the system log says the sandbox registration failed). This is expected for a helper signed to inherit its sandbox, and it is why the bundle check reads the engine's version and commit from the file. Judge the engine only through the app in the Applications folder.
- Observatory's header reads "Stable" in the store build, because the store version runs the stable engine. That is correct.

## Validate, upload, submit

1. **Package:** `cargo xtask release app-store-pkg --app-bundle PATH --installer-identity "$INSTALLER_IDENTITY"`. The identity's name starts with `3rd Party Mac Developer Installer:`.
2. **Validate:** `cargo xtask release app-store-upload --package PATH --api-key KEY_ID --api-issuer ISSUER_ID`. This sends nothing permanent and takes about a minute. Run it early and often: it finds in a minute what reading code does not.
3. **Ask, then upload:** the same command with `--upload`. Apple then processes the build for ten to thirty minutes before it can be selected.
4. **On the record:** select the build on the version, write "What's New" (not asked for on a first version), check the review notes still describe the build.
5. **Ask, then submit.** Keep release on manual.

The API key's private half is a `.p8` file in `~/.appstoreconnect/private_keys/`. The Key ID is in the file's name; the Issuer ID is on the Users and Access, Integrations page. Never print the private key, copy it, or write the Key ID or Issuer ID into the repository.

### What Apple's validation has said so far

| Message | Meaning | Answer |
| --- | --- | --- |
| Error 90296 | A helper that only inherits its sandbox must also declare App Sandbox | The inherited entitlements carry both keys |
| Warning 90889 | No provisioning profile in the bundle | Expected. It only matters for TestFlight (aosama/astronomical#2184) |
| A refused Mac app with no category | `LSApplicationCategoryType` missing | The main app names Productivity, matching the record |
| Held for the encryption questions | No export declaration in the build | The store build declares `ITSAppUsesNonExemptEncryption` as false |

### Replacing a build that is waiting for review

Each step marked "ask" waits for the account holder. One yes to "replace the build and resubmit" covers the three asks below; a yes to less does not.

1. Build, test on screen, package and validate the new build.
2. Ask, then upload it, and wait until App Store Connect lists the build as valid. Doing this before the next step keeps the version in the review queue while Apple processes the build; on 2026-11-14 the version was out of the queue for about two minutes.
3. Ask, then remove the version from review.
4. Select the new build on the version. Update the review notes if entitlements or behaviour changed.
5. Ask, then submit again, and read the record back: status, attached build, notes, release type.

A build number can be used once per version, so the new build must come from a later commit.

Through the App Store Connect API these are: `PATCH reviewSubmissions/{id}` with `canceled: true` to remove from review; `PATCH appStoreVersions/{id}/relationships/build` to select the build; `PATCH appStoreReviewDetails/{id}` for the notes; then `POST reviewSubmissions` (platform `MAC_OS`), `POST reviewSubmissionItems` with the version, and `PATCH reviewSubmissions/{id}` with `submitted: true`.

Apple emails the account holder at each step. Two of them read worse than they are:

- **"Developer Rejected"** means the developer removed the version from review. It is not a decision by App Review.
- **A delivery notice with ITMS-90889** is the TestFlight warning from the table above.

## When Apple answers

Apple's verdict goes to the account holder by email; the record's status shows it too.

- **Approved:** nothing is public yet, because release is manual. Install the approved build from the store on a Mac account that has never run the store version, reach a first reply, and try the on-screen list above. Then ask whether to release.
- **Rejected:** copy each reason onto the submission issue in Apple's own words, open one issue per reason, fix, and go through the replace steps. Reply to the reviewer in the Resolution Center only with the account holder's yes.

Tracked in aosama/astronomical#2201.

## Working on the App Store Connect record

The account holder signs in themself, in the browser pane; then the pages can be read and filled.

- **Typing into a field appends** to what is there, and select-all from the keyboard does not select. Focus and select the field first, then type.
- **Reloading a page discards unsaved edits**, including ones the account holder just made. When they say they changed something, read the page before navigating.
- **The browser pane cannot attach files.** Upload and reorder screenshots through the App Store Connect API instead: reserve an `appScreenshots` entry in the version's `appScreenshotSets`, send the bytes to the address it returns, commit it with its checksum, then set the order on the set's relationship.
- **A token for the API** comes from `xcrun altool --generate-jwt --apiKey KEY_ID --apiIssuer ISSUER_ID`. Reading the record through the API is the quickest way to learn its true state.
- **Review notes cannot be saved without a review contact.** The contact is the account holder's own name, phone and email; they give them, and they are never written anywhere else.
- Saving availability asks for confirmation of the country list. Count it: 148.

Screenshots are kept outside version control, in the ignored `private/store-screenshots/` folder: raw captures and the three captioned finals. Take new ones only when the app looks different, from the store build, at 2880 by 1800.

## Before adding to the listing or the build

- A new promise in the listing: do it on screen in the sandboxed build first.
- A new entitlement: add a sentence to the review notes saying what the person can do because of it.
- A feature that sends anything off the Mac: the privacy label, the privacy page and the privacy manifest (`scripts/internal/privacy/PrivacyInfo.xcprivacy`) change in the same release.
- **Problem reports are switched off in the store build** (issue #2413). The direct download can offer a problem report and send it to the developers when the person chooses; the store build has no report desk, serves no `/v1/problem-reports` route and hides the sharing settings, because the engine sees that it keeps its state in the app container (`problem_report_desk_for` in `apps/daemon/src/application.rs`). `cargo test -p astronomical-daemon --lib application_problem_report` and `--lib config_document` guard this. Before removing the guard, change the privacy manifest, the App Privacy answers and the privacy page in that same release, and tell App Review that reports are off until the person turns them on.
- A new model in the catalog: check its licence allows redistribution, or the content-rights answer no longer holds.

## Open work

Read the umbrella issue for the current state. When this was written, every promise in the table above had been tried in the sandbox, version 0.2.105 was waiting for review, and what remained was provisioning profiles for TestFlight, which need the developer account and are not needed for review.
