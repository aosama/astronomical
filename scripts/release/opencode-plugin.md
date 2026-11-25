# Releasing the OpenCode plugin

The OpenCode plugin in `integrations/opencode` is published to npm as `opencode-astronomical`. It is released apart from the app: its own version number, its own tags, no disk image, no Sparkle feed and no GitHub Release. Nothing here touches the app's release, and the app's release procedure (the `sparkle_release` skill) never touches the plugin.

## What a release is

- A version in `integrations/opencode/package.json`, `MAJOR.MINOR.PATCH`.
- A `## MAJOR.MINOR.PATCH` section in `integrations/opencode/CHANGELOG.md` saying what changed and which OpenCode release it was checked against.
- A tag named `opencode-plugin-vMAJOR.MINOR.PATCH` on a commit of `main` that holds both.

Pushing the tag runs `.github/workflows/publish-opencode-plugin.yml`. It checks that the three agree and that the tagged commit is on `main` (`cargo xtask opencode-plugin-release-check --require-on-main TAG`), runs the plugin's checks, and publishes with an npm provenance statement through npm trusted publishing, so no npm token is stored anywhere in the repository.

The workflow never creates a GitHub Release: the README, the website and the app follow `releases/latest` to the app's disk image, and every published GitHub Release is announced on Discord as an app release.

## Before the first release (once)

npm sets up trusted publishing for a package that already exists, so the first version is published by hand from the maintainer's npm account.

1. Create or choose the npm account that will own `opencode-astronomical`, with two-factor authentication on.
2. From a clean checkout of `main`, publish the first version by hand. `npm publish` copies the repository's license into the package while packing:

       cd integrations/opencode
       npm ci
       npm run verify
       npm login
       npm publish --access public

3. On npmjs.com, open the package's settings and add a trusted publisher: GitHub Actions, organization or user `aosama`, repository `astronomical`, workflow `publish-opencode-plugin.yml`, environment `npm`.
4. In the package's publishing access settings, require two-factor authentication and disallow tokens, so only the workflow can publish.
5. In the repository's settings, open Environments, then `npm`, and add yourself as a required reviewer if every publish should wait for your approval.

## Every later release

1. On a branch, update `version` in `integrations/opencode/package.json` and add the matching `## MAJOR.MINOR.PATCH` section to `CHANGELOG.md`, naming the OpenCode release it was checked against. Run the live check in `docs/development.md` against the Development app first.
2. Merge the pull request.
3. Tag the merge commit and push the tag:

       git fetch origin main
       git tag opencode-plugin-vMAJOR.MINOR.PATCH origin/main
       git push origin opencode-plugin-vMAJOR.MINOR.PATCH

4. Watch the workflow run. To run the same check before tagging, check out `origin/main` and run `cargo xtask opencode-plugin-release-check --require-on-main opencode-plugin-vMAJOR.MINOR.PATCH`. If the release check fails, nothing was published: delete the tag, fix the version or changelog in a new pull request, and tag again. An npm version cannot be published twice, so a mistake after publishing needs a new version.
5. Check that `npm view opencode-astronomical version` shows the new version, and that `opencode plugin add opencode-astronomical@MAJOR.MINOR.PATCH` loads it.
