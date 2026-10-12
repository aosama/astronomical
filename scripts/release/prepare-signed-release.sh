#!/usr/bin/env sh

# Convenience entry point for one signed Astronomical Stable release: it
# resolves the documented credential map — exactly one "Developer ID
# Application" code-signing identity, its 10-character Team ID, and the
# "Astronomical Notarization" keychain profile — and then delegates every
# artifact step to scripts/release/prepare-and-publish.sh, which remains the
# only sanctioned release producer. This wrapper performs no artifact work
# itself, so the release procedure, approval gate, and failure rules in
# .opencode/command/release-stable.md all still apply verbatim.
#
# Usage:
#   scripts/release/prepare-signed-release.sh \
#     --tag vX.Y.Z \
#     --notes-file /absolute/path/to/release-notes.md \
#     [--output-directory /absolute/path/to/prepared] \
#     [--publish]
#
# Arguments:
#   --tag TAG              Required. The release tag, for example v0.2.70.
#                          Never inferred; the caller supplies it every run.
#   --notes-file PATH      Required. Path to the release notes Markdown file;
#                          it is copied into the prepared release directory.
#   --output-directory DIR Optional. Resume or review an existing prepared
#                          directory instead of creating a fresh one; with
#                          --publish it selects the exact bytes to publish.
#   --publish              Optional. Publish the reviewed bytes: push the tag
#                          first, because the publisher requires the remote
#                          tag to resolve to the prepared commit.
#
# Behavior and bounds:
#   - Credential resolution and the publisher run in this one shell process;
#     nothing is cached between invocations.
#   - The publisher owns every internal bound, including the 1,200-second
#     notarization wait; allow roughly 25 minutes of wall clock.
#   - The exit code is the publisher's exit code (the script execs it).
#   - Output stays live and unfiltered on purpose: the transcript carries the
#     notarization submission ID needed by later verification steps.

set -eu

RELEASE_TAG=""
RELEASE_NOTES_FILE=""
OUTPUT_DIRECTORY=""
SHOULD_PUBLISH="false"

print_usage() {
    printf '%s\n' "Usage: scripts/release/prepare-signed-release.sh --tag vX.Y.Z --notes-file PATH [--output-directory PATH] [--publish]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag)
            [ "$#" -ge 2 ] || { printf '%s\n' 'Error: --tag requires a value' >&2; print_usage >&2; exit 2; }
            RELEASE_TAG="$2"
            shift 2
            ;;
        --notes-file)
            [ "$#" -ge 2 ] || { printf '%s\n' 'Error: --notes-file requires a value' >&2; print_usage >&2; exit 2; }
            RELEASE_NOTES_FILE="$2"
            shift 2
            ;;
        --output-directory)
            [ "$#" -ge 2 ] || { printf '%s\n' 'Error: --output-directory requires a value' >&2; print_usage >&2; exit 2; }
            OUTPUT_DIRECTORY="$2"
            shift 2
            ;;
        --publish)
            SHOULD_PUBLISH="true"
            shift
            ;;
        *)
            printf '%s\n' "Error: unknown argument $1" >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

[ -n "$RELEASE_TAG" ] || { printf '%s\n' 'Error: --tag is required and is never inferred' >&2; print_usage >&2; exit 2; }
[ -n "$RELEASE_NOTES_FILE" ] || { printf '%s\n' 'Error: --notes-file is required' >&2; print_usage >&2; exit 2; }
[ -f "$RELEASE_NOTES_FILE" ] || { printf '%s\n' "Error: release notes file not found: $RELEASE_NOTES_FILE" >&2; exit 2; }

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

PUBLISHER="scripts/release/prepare-and-publish.sh"
[ -x "$PUBLISHER" ] || { printf '%s\n' "Error: publisher script not found or not executable: $PUBLISHER" >&2; exit 1; }

# The credential map from .opencode/command/release-stable.md, resolved here
# and passed to the publisher in this same shell process. The identity check
# fails closed when the Keychain exposes zero or several Developer ID
# identities, because signing with the wrong one would poison the release.
SIGNING_IDENTITY="$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p')"
IDENTITY_COUNT="$(printf '%s\n' "$SIGNING_IDENTITY" | awk 'NF { count += 1 } END { print count + 0 }')"
[ "$IDENTITY_COUNT" = 1 ] || { printf '%s\n' 'Error: exactly one Developer ID Application identity is required' >&2; exit 1; }
TEAM_ID="$(printf '%s\n' "$SIGNING_IDENTITY" | sed -n 's/.*(\([[:alnum:]]\{10\}\))$/\1/p')"
[ "${#TEAM_ID}" = 10 ] || { printf '%s\n' 'Error: Developer ID identity does not contain a 10-character Team ID' >&2; exit 1; }
NOTARY_PROFILE="Astronomical Notarization"

printf '%s\n' "[prepare-signed-release] resolved credential map; delegating to $PUBLISHER for $RELEASE_TAG"

PUBLISH_ARGS=""
[ "$SHOULD_PUBLISH" = "true" ] && PUBLISH_ARGS="--publish"
exec "$PUBLISHER" \
    --tag "$RELEASE_TAG" \
    --notes-file "$RELEASE_NOTES_FILE" \
    ${OUTPUT_DIRECTORY:+--output-directory "$OUTPUT_DIRECTORY"} \
    $PUBLISH_ARGS \
    --signing-identity "$SIGNING_IDENTITY" \
    --team-id "$TEAM_ID" \
    --notary-profile "$NOTARY_PROFILE"
