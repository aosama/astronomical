#!/usr/bin/env sh

# Renders the recorded step segments and cache reports into one run-summary
# table so every CI run attributes its elapsed time to individual steps
# without opening the hosted action logs.

set -eu

timing_file="${ASTRONOMICAL_CI_TIMING_FILE:-${TMPDIR:-/tmp}/astronomical-ci-step-timings.csv}"
segment_starts_directory="$(dirname -- "$timing_file")/astronomical-ci-step-starts"

measured_rows=""
if [ -f "$timing_file" ]; then
    measured_rows="$(cat "$timing_file")"
fi

aborted_rows=""
measured_segment_count=0
if [ -f "$timing_file" ]; then
    measured_segment_count="$(grep -c '[^[:space:]]' "$timing_file" || true)"
fi
if [ -d "$segment_starts_directory" ]; then
    for start_marker_path in "$segment_starts_directory"/*; do
        [ -e "$start_marker_path" ] || continue
        aborted_rows="$aborted_rows$(basename "$start_marker_path")\n"
    done
fi

measured_table=""
if [ -n "$measured_rows" ]; then
    measured_table="$(printf '%s\n' "$measured_rows" | awk -F, 'NF == 2 { printf "| %s | %s |\n", $1, $2 }')"
fi
aborted_table=""
if [ -n "$aborted_rows" ]; then
    aborted_table="$(printf '%b' "$aborted_rows" | awk -F, 'NF >= 1 && $1 != "" { printf "| %s | aborted |\n", $1 }')"
fi

summary_body=""
add_summary_line() {
    summary_body="$summary_body$1
"
}

add_summary_line '### Step timing attribution'
add_summary_line ''
if [ -n "$measured_table" ] || [ -n "$aborted_table" ]; then
    add_summary_line '| Segment | Elapsed seconds |'
    add_summary_line '| --- | --- |'
    if [ -n "$measured_table" ]; then
        summary_body="$summary_body$measured_table
"
    fi
    if [ -n "$aborted_table" ]; then
        summary_body="$summary_body$aborted_table
"
    fi
else
    add_summary_line 'No timed segments were recorded.'
fi

printf '[ci-timing-summary] measured_segments=%s aborted_segments=%s status=complete\n' \
    "$measured_segment_count" "$(printf '%b' "$aborted_rows" | grep -c '[^[:space:]]' || true)"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n' "$summary_body" >> "$GITHUB_STEP_SUMMARY"
else
    printf '%s\n' "$summary_body"
fi
