#!/usr/bin/env sh

# Workstream 5 of issue #1198: count tests per tier and report the pyramid
# distribution. The distribution is the earliest warning that tests are
# migrating to the expensive tiers, so the committed baseline is compared with
# the live counts and any tier drifting beyond the tolerance fails the step.
# The direction of travel for the pyramid is ~70% unit, ~25% integration,
# ~4% end-to-end, ~1% acceptance; drift is measured against the recorded
# baseline rather than those targets so the alarm fires on migration, not on
# day-one reality.
#
# Counting is static and deterministic so any laptop reproduces it:
# - unit: harness macro invocations under any crate or app source tree,
#   where beside-source test modules live.
# - integration: harness macros plus raw test attributes under tests/
#   directories, excluding helper folders (fixtures, support, common).
# - e2e: test attributes under the root tests/e2e package in files without
#   the real-model fidelity toggle.
# - acceptance: test attributes under the root tests/acceptance package,
#   whose suites load real model artifacts by design, plus test attributes
#   under tests/e2e in files gated by ASTRONOMICAL_REAL_MODEL — the same
#   journeys at real-model fidelity.

set -eu

DRIFT_TOLERANCE_PERCENT=5
BASELINE_RELATIVE_PATH="scripts/ci/tier-pyramid-baseline.json"

UNIT_MACRO_PATTERN='offline_test!\(|offline_tokio_test!\(|offline_tokio_real_clock_test!\(|offline_denied_test!\(|offline_denied_tokio_test!\('
TEST_ATTRIBUTE_PATTERN='^#\[test\]$|^#\[tokio::test'

sum_counts_from_stdin() {
    awk '{ total += $1 } END { print total + 0 }'
}

count_unit() {
    repository_root="$1"
    {
        find "${repository_root}"/crates/*/src -name '*.rs' -type f 2>/dev/null || true
        find "${repository_root}"/apps/*/src -name '*.rs' -type f 2>/dev/null || true
    } | while read -r source_file; do
        rg -c "$UNIT_MACRO_PATTERN" "$source_file" 2>/dev/null || true
    done | sum_counts_from_stdin
}

count_integration() {
    repository_root="$1"
    {
        find "${repository_root}"/crates/*/tests -name '*.rs' -type f 2>/dev/null || true
        find "${repository_root}"/apps/*/tests -name '*.rs' -type f 2>/dev/null || true
    } | grep -v '/e2e/' |
        grep -v '/fixtures/' |
        grep -v '/support/' |
        grep -v '/common/' |
        while read -r source_file; do
            rg -c "$UNIT_MACRO_PATTERN" "$source_file" 2>/dev/null || true
            rg -c "$TEST_ATTRIBUTE_PATTERN" "$source_file" 2>/dev/null || true
        done | sum_counts_from_stdin
}

count_e2e_tier() {
    repository_root="$1"
    tier="$2"
    e2e_tests_directory="${repository_root}/tests/e2e/tests"
    [ -d "$e2e_tests_directory" ] || {
        echo 0
        return
    }
    total=0
    for source_file in "$e2e_tests_directory"/*.rs; do
        [ -f "$source_file" ] || continue
        if [ "$tier" = "acceptance" ]; then
            grep -q 'ASTRONOMICAL_REAL_MODEL' "$source_file" || continue
        else
            grep -q 'ASTRONOMICAL_REAL_MODEL' "$source_file" && continue
        fi
        file_count="$(rg -c "$TEST_ATTRIBUTE_PATTERN" "$source_file" 2>/dev/null | sum_counts_from_stdin)"
        total=$((total + file_count))
    done
    echo "$total"
}

count_acceptance_package() {
    repository_root="$1"
    acceptance_tests_directory="${repository_root}/tests/acceptance/tests"
    [ -d "$acceptance_tests_directory" ] || {
        echo 0
        return
    }
    find "$acceptance_tests_directory" -name '*.rs' -type f 2>/dev/null |
        grep -v '/fixtures/' |
        grep -v '/support/' |
        grep -v '/common/' |
        while read -r source_file; do
            rg -c "$UNIT_MACRO_PATTERN" "$source_file" 2>/dev/null || true
            rg -c "$TEST_ATTRIBUTE_PATTERN" "$source_file" 2>/dev/null || true
        done | sum_counts_from_stdin
}

print_usage_and_exit() {
    printf 'usage: report-tier-pyramid.sh [--repository-root PATH] [--update-baseline]\n' >&2
    exit 2
}

main() {
    repository_root=""
    update_baseline=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --repository-root)
                [ "$#" -ge 2 ] || print_usage_and_exit
                repository_root="$2"
                shift 2
                ;;
            --update-baseline)
                update_baseline=1
                shift
                ;;
            *)
                print_usage_and_exit
                ;;
        esac
    done
    if [ -z "$repository_root" ]; then
        repository_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
    fi

    unit_count="$(count_unit "$repository_root")"
    integration_count="$(count_integration "$repository_root")"
    e2e_count="$(count_e2e_tier "$repository_root" e2e)"
    acceptance_count="$(
        (count_e2e_tier "$repository_root" acceptance; count_acceptance_package "$repository_root") | sum_counts_from_stdin
    )"
    total_count=$((unit_count + integration_count + e2e_count + acceptance_count))

    printf '%s\n' '[tier-pyramid] status=start mode=report'
    awk -v unit="$unit_count" -v integration="$integration_count" \
        -v e2e="$e2e_count" -v acceptance="$acceptance_count" \
        -v total="$total_count" 'BEGIN {
        unit_share = total > 0 ? unit * 100 / total : 0
        integration_share = total > 0 ? integration * 100 / total : 0
        e2e_share = total > 0 ? e2e * 100 / total : 0
        acceptance_share = total > 0 ? acceptance * 100 / total : 0
        printf "unit=%d share=%.1f%%\n", unit, unit_share
        printf "integration=%d share=%.1f%%\n", integration, integration_share
        printf "e2e=%d share=%.1f%%\n", e2e, e2e_share
        printf "acceptance=%d share=%.1f%%\n", acceptance, acceptance_share
        printf "total=%d\n", total
    }'

    baseline_path="${repository_root}/${BASELINE_RELATIVE_PATH}"
    if [ "$update_baseline" -eq 1 ]; then
        printf '%s\n' '[tier-pyramid] status=baseline-update'
        mkdir -p "$(dirname "$baseline_path")"
        {
            printf '{\n'
            printf '  "unit": %d,\n' "$unit_count"
            printf '  "integration": %d,\n' "$integration_count"
            printf '  "e2e": %d,\n' "$e2e_count"
            printf '  "acceptance": %d\n' "$acceptance_count"
            printf '}\n'
        } > "$baseline_path"
        printf '%s\n' "[tier-pyramid] status=baseline-written path=${BASELINE_RELATIVE_PATH}"
        printf '%s\n' '[tier-pyramid] status=success'
        return
    fi

    if [ ! -f "$baseline_path" ]; then
        printf '%s\n' '[tier-pyramid] status=no-baseline hint=record-one-with---update-baseline'
        printf '%s\n' '[tier-pyramid] status=success'
        return
    fi

    baseline_unit="$(sed -n 's/.*"unit": \([0-9]*\).*/\1/p' "$baseline_path" | head -1)"
    baseline_integration="$(sed -n 's/.*"integration": \([0-9]*\).*/\1/p' "$baseline_path" | head -1)"
    baseline_e2e="$(sed -n 's/.*"e2e": \([0-9]*\).*/\1/p' "$baseline_path" | head -1)"
    baseline_acceptance="$(sed -n 's/.*"acceptance": \([0-9]*\).*/\1/p' "$baseline_path" | head -1)"
    baseline_total=$((baseline_unit + baseline_integration + baseline_e2e + baseline_acceptance))
    if [ "$baseline_total" -eq 0 ] || [ "$total_count" -eq 0 ]; then
        printf '%s\n' '[tier-pyramid] status=empty-distribution hint=record-one-with---update-baseline'
        printf '%s\n' '[tier-pyramid] status=success'
        return
    fi

    drifted_tier="$(
        awk -v tolerance="$DRIFT_TOLERANCE_PERCENT" -v total="$total_count" \
            -v baseline_total="$baseline_total" \
            -v unit="$unit_count" -v integration="$integration_count" \
            -v e2e="$e2e_count" -v acceptance="$acceptance_count" \
            -v baseline_unit="$baseline_unit" -v baseline_integration="$baseline_integration" \
            -v baseline_e2e="$baseline_e2e" -v baseline_acceptance="$baseline_acceptance" 'BEGIN {
            tier_names[1] = "unit"; tier_names[2] = "integration"
            tier_names[3] = "e2e"; tier_names[4] = "acceptance"
            current["unit"] = unit; current["integration"] = integration
            current["e2e"] = e2e; current["acceptance"] = acceptance
            baseline["unit"] = baseline_unit; baseline["integration"] = baseline_integration
            baseline["e2e"] = baseline_e2e; baseline["acceptance"] = baseline_acceptance
            for (tier_index = 1; tier_index <= 4; tier_index += 1) {
                tier = tier_names[tier_index]
                drift = current[tier] * 100 / total - baseline[tier] * 100 / baseline_total
                if (drift < 0) {
                    drift = -drift
                }
                if (drift > tolerance) {
                    printf "%s drifted %.1f percentage points from the recorded baseline", tier, drift
                    exit 0
                }
            }
        }')"
    if [ -n "$drifted_tier" ]; then
        printf '%s\n' "[tier-pyramid] status=drift-detected tier=${drifted_tier}"
        printf '%s\n' \
            '[tier-pyramid] hint=if-this-migration-is-intended-refresh-scripts/ci/tier-pyramid-baseline.json-with---update-baseline-in-a-reviewed-pull-request'
        exit 1
    fi

    printf '%s\n' '[tier-pyramid] status=success reason=within-tolerance'
}

main "$@"
