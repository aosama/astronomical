#!/usr/bin/env sh

# Reports one cache owner's operation independently so hosted action logs expose
# that owner's duration, key, classification, and compressed transfer bytes.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

require_environment_value() {
    environment_name="$1"
    environment_value="$2"
    [ -n "$environment_value" ] || {
        print_error "${environment_name} is required"
        exit 2
    }
}

validate_unsigned_integer() {
    integer_name="$1"
    integer_value="$2"
    case "$integer_value" in
        ''|*[!0-9]*)
            print_error "${integer_name} must be an unsigned integer"
            exit 2
            ;;
    esac
}

classify_restore() {
    if [ "$CACHE_STEP_OUTCOME" != "success" ]; then
        cache_classification="error"
        return
    fi
    case "${CACHE_HIT:-}" in
        true)
            [ "${CACHE_MATCHED_KEY:-}" = "$CACHE_PRIMARY_KEY" ] || {
                print_error "a primary hit must match CACHE_PRIMARY_KEY"
                exit 2
            }
            cache_classification="primary"
            ;;
        false|'')
            if [ -z "${CACHE_MATCHED_KEY:-}" ]; then
                cache_classification="miss"
            elif [ "$CACHE_MATCHED_KEY" = "$CACHE_PRIMARY_KEY" ]; then
                print_error "a non-primary restore cannot match CACHE_PRIMARY_KEY"
                exit 2
            else
                cache_classification="fallback"
            fi
            ;;
        *)
            print_error "CACHE_HIT must be true, false, or empty"
            exit 2
            ;;
    esac
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "report-build-cache-restoration.sh does not accept arguments"
        exit 2
    fi
    require_environment_value CACHE_OWNER "${CACHE_OWNER:-}"
    require_environment_value CACHE_OPERATION "${CACHE_OPERATION:-}"
    require_environment_value CACHE_STEP_OUTCOME "${CACHE_STEP_OUTCOME:-}"
    require_environment_value CACHE_PRIMARY_KEY "${CACHE_PRIMARY_KEY:-}"
    require_environment_value CACHE_STARTED_AT_EPOCH_SECONDS "${CACHE_STARTED_AT_EPOCH_SECONDS:-}"
    require_environment_value CACHE_FINISHED_AT_EPOCH_SECONDS "${CACHE_FINISHED_AT_EPOCH_SECONDS:-}"
    case "$CACHE_OWNER" in
        *[!a-z0-9-]*|-*|*-) print_error "CACHE_OWNER must use lowercase words separated by hyphens"; exit 2 ;;
    esac
    case "$CACHE_OPERATION" in
        restore) classify_restore ;;
        save)
            if [ "$CACHE_STEP_OUTCOME" = "success" ]; then
                cache_classification="published"
            else
                cache_classification="error"
            fi
            ;;
        *) print_error "CACHE_OPERATION must be restore or save"; exit 2 ;;
    esac
    validate_unsigned_integer CACHE_STARTED_AT_EPOCH_SECONDS "$CACHE_STARTED_AT_EPOCH_SECONDS"
    validate_unsigned_integer CACHE_FINISHED_AT_EPOCH_SECONDS "$CACHE_FINISHED_AT_EPOCH_SECONDS"
    [ "$CACHE_FINISHED_AT_EPOCH_SECONDS" -ge "$CACHE_STARTED_AT_EPOCH_SECONDS" ] || {
        print_error "cache completion time precedes its start time"
        exit 2
    }
    elapsed_seconds="$((CACHE_FINISHED_AT_EPOCH_SECONDS - CACHE_STARTED_AT_EPOCH_SECONDS))"
    matched_key="${CACHE_MATCHED_KEY:-none}"
    printf '[build-cache] owner=%s operation=%s classification=%s elapsed_seconds=%s primary_key=%s matched_key=%s\n' \
        "$CACHE_OWNER" "$CACHE_OPERATION" "$cache_classification" "$elapsed_seconds" \
        "$CACHE_PRIMARY_KEY" "$matched_key"

    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            printf '### Build cache: %s %s\n\n' "$CACHE_OWNER" "$CACHE_OPERATION"
            printf '%s\n' "- Classification: \`${cache_classification}\`"
            printf '%s\n' "- Elapsed seconds: \`${elapsed_seconds}\`"
            printf '%s\n' "- Primary key: \`${CACHE_PRIMARY_KEY}\`"
            printf '%s\n' "- Matched key: \`${matched_key}\`"
            printf '\nThe named cache action log records this owner%s exact compressed transfer bytes.\n' "'s"
        } >> "$GITHUB_STEP_SUMMARY"
    fi
}

main "$@"
