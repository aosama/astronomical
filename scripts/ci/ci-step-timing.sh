#!/usr/bin/env sh

# Records named CI step segments so each verification job can publish one
# step-timing table in the run summary. Begin/end wrap a step's run block; the
# begin marker also makes an interrupted step visible as an aborted segment
# instead of silently disappearing from the attribution table.

set -eu

print_error() {
    printf '%s\n' "Error: $1" >&2
}

timing_file_path() {
    printf '%s' "${ASTRONOMICAL_CI_TIMING_FILE:-${TMPDIR:-/tmp}/astronomical-ci-step-timings.csv}"
}

segment_starts_directory() {
    printf '%s' "$(dirname -- "$(timing_file_path)")/astronomical-ci-step-starts"
}

validate_segment_label() {
    case "$1" in
        ''|-*|*[!a-z0-9-]*|*-)
            print_error "step label must be lowercase words separated by hyphens: $1"
            exit 2
            ;;
    esac
}

usage() {
    print_error "usage: $0 begin <step-label> | $0 end <step-label>"
    exit 2
}

[ "$#" -eq 2 ] || usage
timing_operation="$1"
segment_label="$2"
validate_segment_label "$segment_label"

case "$timing_operation" in
    begin)
        starts_directory="$(segment_starts_directory)"
        mkdir -p "$starts_directory"
        printf '%s\n' "$(date +%s)" > "$starts_directory/$segment_label"
        printf '[ci-step-timing] segment=%s status=begin\n' "$segment_label"
        ;;
    end)
        start_marker_path="$(segment_starts_directory)/$segment_label"
        [ -f "$start_marker_path" ] || {
            print_error "no begin marker recorded for segment: $segment_label"
            exit 2
        }
        started_at_epoch_seconds="$(tr -d '[:space:]' < "$start_marker_path")"
        case "$started_at_epoch_seconds" in
            ''|*[!0-9]*)
                print_error "begin marker holds a non-numeric start time: $segment_label"
                exit 2
                ;;
        esac
        current_at_epoch_seconds="$(date +%s)"
        elapsed_seconds=$(( current_at_epoch_seconds - started_at_epoch_seconds ))
        printf '%s,%s\n' "$segment_label" "$elapsed_seconds" >> "$(timing_file_path)"
        rm -f "$start_marker_path"
        printf '[ci-step-timing] segment=%s status=end elapsed_seconds=%s\n' "$segment_label" "$elapsed_seconds"
        ;;
    *)
        usage
        ;;
esac
