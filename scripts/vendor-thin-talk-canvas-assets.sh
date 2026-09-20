#!/usr/bin/env bash
#
# Vendors the pinned third-party web assets the Thin Talk conversation canvas
# renders with. The canvas must work with no network at all, so every asset is
# committed to the repository and served to the web view through a private
# scheme handler instead of a CDN.
#
# Usage: scripts/vendor-thin-talk-canvas-assets.sh [--verify-only]
#
# Exit codes:
#   0 — every asset is present and matches its pinned digest
#   1 — a download failed, or a file does not match its pinned digest

set -euo pipefail

readonly REPOSITORY_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
readonly VENDOR_DIRECTORY="${REPOSITORY_ROOT}/apps/thin-talk/Sources/ThinTalkCanvas/Resources/web/vendor"
readonly DOWNLOAD_URL_PREFIX="https://cdn.jsdelivr.net/npm"

VERIFY_ONLY="false"

# name|package path|sha256|kind (script|style|license|notice)
readonly PINNED_ASSETS=(
    "marked.min.js|marked@15.0.7/marked.min.js|934e3e36e9e2da0afb1a6e75075bb0f09af05293a844e84a7477ef40911c349a|script"
    "purify.min.js|dompurify@3.2.4/dist/purify.min.js|8eb41b658831fab175fad9bcd00fcb2d84e0ed3a25a55053d4ecd4444b8b43a0|script"
    "morphdom-umd.min.js|morphdom@2.7.7/dist/morphdom-umd.min.js|ad1aaf5441eb2798b99dd03a41bea26562cd634dfc9845d3b8c5fbce560a6bac|script"
    "highlight.min.js|@highlightjs/cdn-assets@11.11.1/highlight.min.js|c4a399dd6f488bc97a3546e3476747b3e714c99c57b9473154c6fb8d259b9381|script"
    "highlight-github-dark.min.css|@highlightjs/cdn-assets@11.11.1/styles/github-dark.min.css|9f208d022102b1d0c7aebfecd8e42ca7997d5de636649d2b31ea63093d809019|style",
    "katex/katex.min.js|katex@0.16.22/dist/katex.min.js|e8d885505949f3a5f4abdd5dd0d53696bd1371ad26ffbf4f310dcd77c8cdae89|script",
    "katex/katex.min.css|katex@0.16.22/dist/katex.min.css|19095127357ed6d29fe0a63a6b000c913a89f7f1963b765dd3715e97c9852e75|style",
    "mermaid.min.js|mermaid@11.12.2/dist/mermaid.min.js|d0830a6c05546e9edb8fe20a8f545f3e0dc7c4c3134d584bad9c13a99d7a71e0|script"
)

# KaTeX renders its fonts through relative URLs, and serving them as plain assets
# through the scheme handler keeps the stylesheet unchanged and auditable.
readonly PINNED_FONTS=(
    "katex/fonts/KaTeX_AMS-Regular.woff2|0cdd387c9590a1a9f9794560022dbb59654a7d86f187aa0c81495ad42d3a7308"
    "katex/fonts/KaTeX_Caligraphic-Bold.woff2|de7701e42cf1f4cf0b766c03fb27977207eee2f4fd5d76fa82188406da43ea4c"
    "katex/fonts/KaTeX_Caligraphic-Regular.woff2|5d53e70ad607c2352162dec9e0923fb54ecdafaccbf604cd8dcf7d00facb989b"
    "katex/fonts/KaTeX_Fraktur-Bold.woff2|74444efd593c005e3f4573b44524704c0af0a937fe911cca9e94068d0d140d3f"
    "katex/fonts/KaTeX_Fraktur-Regular.woff2|51814d270d06ff0255dba0799994fa4d8c84d11f09951d47595f4abb1f3602dc"
    "katex/fonts/KaTeX_Main-Bold.woff2|0f60d1b897938ec918c8ce073092411baf9438f6739465693ff18b0f9d20b021"
    "katex/fonts/KaTeX_Main-BoldItalic.woff2|99cd42a3c072d918f2f44984a807cf7aa16e13545fd0875fc07c6c65f99e715b"
    "katex/fonts/KaTeX_Main-Italic.woff2|97479ca6cce906abc961ecac96faa5f9ca2e61b8e7670d475826bcdee9a7c267"
    "katex/fonts/KaTeX_Main-Regular.woff2|c2342cd8b869e01752a9321dc17213fc40d4d04c79688c1d43f2cf316abd7866"
    "katex/fonts/KaTeX_Math-BoldItalic.woff2|dc47344dbb6cb5b655c8460d561f4df5f501b90c804ad3c6cec65fe322351ab1"
    "katex/fonts/KaTeX_Math-Italic.woff2|7af58c5ec8f132a2ddde9027c6d7814decce4d3b822a11192a42a20e2e973264"
    "katex/fonts/KaTeX_SansSerif-Bold.woff2|e99ae51144bf1232efcc1bfe5add36262c6866b0faab24fa75740e1b98577a62"
    "katex/fonts/KaTeX_SansSerif-Italic.woff2|00b26ac825e2095056396e0553b8ac26d3f8ad158c3826e28b4c45b385c4714a"
    "katex/fonts/KaTeX_SansSerif-Regular.woff2|68e8c73ef42afd3ccec58bf0fba302cce448938e7fc020a5e31f8a952eee1342"
    "katex/fonts/KaTeX_Script-Regular.woff2|036d4e95149b69ff9bcc0cd55771efeb25ffa3947293e69acd78d5ac328c684b"
    "katex/fonts/KaTeX_Size1-Regular.woff2|6b47c40166b6dbe21a5dfca7718413f2147fd2399be1ba605d8ad39cedf25dfe"
    "katex/fonts/KaTeX_Size2-Regular.woff2|d04c54219f9eaec6d4d4fd42dfb28785975a4794d6b2fc71e566b9cd6db842dd"
    "katex/fonts/KaTeX_Size3-Regular.woff2|73d591271b1604960cb10bb90fee021670af7297017e0e98480b332d11f51995"
    "katex/fonts/KaTeX_Size4-Regular.woff2|a4af7d414440a1c1790825cfb700cf9cf43b0f2c4b04f0ebc523011ad9853ec0"
    "katex/fonts/KaTeX_Typewriter-Regular.woff2|71d517d67827787cfabdf186914cc3358eda539e37931941f2b2fd4a21f68c0b"
)

readonly PINNED_LICENSES=(
    "LICENSE-marked.md|marked@15.0.7/LICENSE.md"
    "LICENSE-dompurify.txt|dompurify@3.2.4/LICENSE"
    "LICENSE-morphdom.txt|morphdom@2.7.7/LICENSE"
    "LICENSE-highlightjs.txt|@highlightjs/cdn-assets@11.11.1/LICENSE"
    "katex/LICENSE-katex.txt|katex@0.16.22/LICENSE"
    "LICENSE-mermaid.txt|mermaid@11.12.2/LICENSE"
)

print_progress() {
    printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1"
}

print_error() {
    printf '%s\n' "Error: $1" >&2
}

compute_digest() {
    shasum -a 256 "$1" | awk '{print $1}'
}

file_digest_matches() {
    local asset_path="$1"
    local pinned_digest="$2"
    [ -f "${asset_path}" ] || return 1
    [ "$(compute_digest "${asset_path}")" = "${pinned_digest}" ]
}

download_asset() {
    local destination_path="$1"
    local package_path="$2"
    local temporary_path="${destination_path}.download"
    if ! curl --fail --silent --show-error --location \
        --output "${temporary_path}" "${DOWNLOAD_URL_PREFIX}/${package_path}"; then
        rm -f "${temporary_path}"
        return 1
    fi
    mv "${temporary_path}" "${destination_path}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --verify-only)
            VERIFY_ONLY="true"
            shift
            ;;
        --help|-h)
            printf '%s\n' "Usage: scripts/vendor-thin-talk-canvas-assets.sh [--verify-only]"
            exit 0
            ;;
        *)
            print_error "unrecognized argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${VENDOR_DIRECTORY}"

print_progress "operation=vendor-canvas-assets status=start directory=${VENDOR_DIRECTORY}"

mismatched_assets=0
for pinned_asset in "${PINNED_ASSETS[@]}"; do
    IFS='|' read -r asset_name package_path pinned_digest asset_kind <<< "${pinned_asset}"
    asset_path="${VENDOR_DIRECTORY}/${asset_name}"

    if [ "${VERIFY_ONLY}" = "false" ] && ! file_digest_matches "${asset_path}" "${pinned_digest}"; then
        print_progress "operation=vendor-canvas-assets status=download asset=${asset_name} kind=${asset_kind}"
        if ! download_asset "${asset_path}" "${package_path}"; then
            print_error "could not download ${asset_name} from ${package_path}"
            exit 1
        fi
    fi

    actual_digest="$(compute_digest "${asset_path}" 2>/dev/null || true)"
    if [ "${actual_digest}" = "${pinned_digest}" ]; then
        print_progress "operation=vendor-canvas-assets status=verified asset=${asset_name} bytes=$(wc -c < "${asset_path}" | tr -d ' ')"
    else
        print_error "${asset_name} digest mismatch: expected ${pinned_digest}, found ${actual_digest:-missing}"
        mismatched_assets=$((mismatched_assets + 1))
    fi
done

for pinned_license in "${PINNED_LICENSES[@]}"; do
    IFS='|' read -r license_name package_path <<< "${pinned_license}"
    license_path="${VENDOR_DIRECTORY}/${license_name}"
    if [ ! -s "${license_path}" ] && [ "${VERIFY_ONLY}" = "false" ]; then
        print_progress "operation=vendor-canvas-assets status=download asset=${license_name} kind=license"
        if ! download_asset "${license_path}" "${package_path}"; then
            print_error "could not download ${license_name} from ${package_path}"
            exit 1
        fi
    fi
    if [ -s "${license_path}" ]; then
        print_progress "operation=vendor-canvas-assets status=verified asset=${license_name} bytes=$(wc -c < "${license_path}" | tr -d ' ')"
    else
        print_error "${license_name} is missing"
        mismatched_assets=$((mismatched_assets + 1))
    fi
done

for pinned_font in "${PINNED_FONTS[@]}"; do
    IFS='|' read -r font_name pinned_digest <<< "${pinned_font}"
    font_path="${VENDOR_DIRECTORY}/${font_name}"
    font_package="katex@0.16.22/dist/fonts/$(basename "${font_name}")"
    if [ "${VERIFY_ONLY}" = "false" ] && ! file_digest_matches "${font_path}" "${pinned_digest}"; then
        print_progress "operation=vendor-canvas-assets status=download asset=${font_name} kind=font"
        if ! download_asset "${font_path}" "${font_package}"; then
            print_error "could not download ${font_name} from ${font_package}"
            exit 1
        fi
    fi
    if file_digest_matches "${font_path}" "${pinned_digest}"; then
        print_progress "operation=vendor-canvas-assets status=verified asset=${font_name} bytes=$(wc -c < "${font_path}" | tr -d ' ')"
    else
        print_error "${font_name} digest mismatch: expected ${pinned_digest}, found $(compute_digest "${font_path}" 2>/dev/null || echo missing)"
        mismatched_assets=$((mismatched_assets + 1))
    fi
done

if [ "${mismatched_assets}" -gt 0 ]; then
    print_progress "operation=vendor-canvas-assets status=failed mismatched=${mismatched_assets}"
    exit 1
fi

print_progress "operation=vendor-canvas-assets status=success assets=$((${#PINNED_ASSETS[@]} + ${#PINNED_FONTS[@]} + ${#PINNED_LICENSES[@]}))"