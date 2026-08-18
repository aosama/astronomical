#!/usr/bin/env sh

# Classifies actions/cache restoration so timing evidence always distinguishes
# a reusable build from a fallback or cold native rebuild.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "report-build-cache-restoration.sh does not accept arguments"
        exit 2
    fi
    [ -n "${CACHE_PRIMARY_KEY:-}" ] || {
        print_error "CACHE_PRIMARY_KEY is required"
        exit 2
    }

    case "${CACHE_HIT:-}:${CACHE_MATCHED_KEY:-}" in
        true:*) cache_classification=primary ;;
        false:?*) cache_classification=fallback ;;
        :?*) cache_classification=fallback ;;
        false:|:) cache_classification=miss ;;
        *)
            print_error "CACHE_HIT must be true, false, or empty"
            exit 2
            ;;
    esac
    printf '[build-cache] classification=%s primary_key=%s matched_key=%s\n' \
        "$cache_classification" "$CACHE_PRIMARY_KEY" "${CACHE_MATCHED_KEY:-none}"

    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            printf '### Build cache restoration\n\n'
            printf -- "- Classification: \`%s\`\n" "$cache_classification"
            printf -- "- Primary key: \`%s\`\n" "$CACHE_PRIMARY_KEY"
            printf -- "- Matched key: \`%s\`\n" "${CACHE_MATCHED_KEY:-none}"
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

main "$@"
