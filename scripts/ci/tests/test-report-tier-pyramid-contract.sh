#!/usr/bin/env sh

# Pins the tier-pyramid report contract: deterministic tier counting from a
# repository tree, report format, drift alerting against the committed
# baseline, and the explicit --update-baseline escape hatch.

set -eu

SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe pyramid-contract sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

write_unit_fixture() {
    fixture_root="$1"
    mkdir -p "${fixture_root}/crates/example/src" \
        "${fixture_root}/crates/example/tests/offline" \
        "${fixture_root}/crates/example/tests/support" \
        "${fixture_root}/tests/e2e/tests"
    cat > "${fixture_root}/crates/example/src/example_tests.rs" <<'EOF'
use astronomical_test_harness::offline_test;
use astronomical_test_harness::offline_tokio_test;

offline_test!(first_unit_case, {
    assert_eq!(1 + 1, 2);
});

offline_tokio_test!(second_unit_case, {
    assert_eq!(2 + 2, 4);
});
EOF
    cat > "${fixture_root}/crates/example/tests/offline/journeys.rs" <<'EOF'
use astronomical_test_harness::offline_test;

offline_test!(first_integration_case, {
    assert_eq!(1, 1);
});

#[test]
fn second_integration_case() {
    assert_eq!(1, 1);
}
EOF
    cat > "${fixture_root}/crates/example/tests/support/helper.rs" <<'EOF'
#[test]
fn helper_must_not_count_as_a_test() {
    assert_eq!(1, 1);
}
EOF
    cat > "${fixture_root}/tests/e2e/tests/product_journey.rs" <<'EOF'
#[tokio::test]
async fn one_end_to_end_case() {
    assert_eq!(1, 1);
}
EOF
    cat > "${fixture_root}/tests/e2e/tests/real_model_journey.rs" <<'EOF'
// ASTRONOMICAL_REAL_MODEL=1 drives the byte-identical request flow against real artifacts.
#[tokio::test]
async fn one_acceptance_case() {
    assert_eq!(1, 1);
}
EOF
}

main() {
    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
    pyramid_script="${repository_root}/scripts/ci/report-tier-pyramid.sh"
    [ -x "$pyramid_script" ] || {
        print_error "scripts/ci/report-tier-pyramid.sh must exist and be executable"
        exit 1
    }

    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-pyramid-contract.XXXXXX")"
    fixture_root="${SANDBOX_DIRECTORY}/repository"
    mkdir -p "$fixture_root"
    write_unit_fixture "$fixture_root"

    printf '%s\n' '[tier-pyramid-contract] case=counts-each-tier-from-the-tree status=start'
    first_report="$("$pyramid_script" --repository-root "$fixture_root" 2>&1)" || {
        print_error "reporting against a fresh fixture tree without a baseline must guide, not fail: $first_report"
        exit 1
    }
    printf '%s\n' "$first_report" | grep -q "unit=2" || {
        print_error "unit tier must count 2 harness macro invocations, got: $first_report"
        exit 1
    }
    printf '%s\n' "$first_report" | grep -q "integration=2" || {
        print_error "integration tier must count 2 tests and exclude the support helper, got: $first_report"
        exit 1
    }
    printf '%s\n' "$first_report" | grep -q "e2e=1" || {
        print_error "e2e tier must count 1 journey, got: $first_report"
        exit 1
    }
    printf '%s\n' "$first_report" | grep -q "acceptance=1" || {
        print_error "acceptance tier must count the real-model gated journey, got: $first_report"
        exit 1
    }

    printf '%s\n' '[tier-pyramid-contract] case=record-baseline-then-pass-within-tolerance status=start'
    "$pyramid_script" --repository-root "$fixture_root" --update-baseline > /dev/null 2>&1 || {
        print_error "--update-baseline must record the fixture distribution"
        exit 1
    }
    [ -f "${fixture_root}/scripts/ci/tier-pyramid-baseline.json" ] || {
        print_error "--update-baseline must write the baseline into scripts/ci/"
        exit 1
    }
    within_tolerance_report="$("$pyramid_script" --repository-root "$fixture_root" 2>&1)" || {
        print_error "reporting must pass when no tier drifted from the baseline: $within_tolerance_report"
        exit 1
    }

    printf '%s\n' '[tier-pyramid-contract] case=fails-naming-the-drifted-tier status=start'
    cat > "${fixture_root}/crates/example/tests/offline/more_journeys.rs" <<'EOF'
offline_test!(third_integration_case, {
    assert_eq!(1, 1);
});
offline_test!(fourth_integration_case, {
    assert_eq!(1, 1);
});
EOF
    drift_exit_status=0
    drift_report="$("$pyramid_script" --repository-root "$fixture_root" 2>&1)" || drift_exit_status=$?
    [ "$drift_exit_status" -ne 0 ] || {
        print_error "reporting must fail when integration grows beyond the tolerance"
        exit 1
    }
    printf '%s\n' "$drift_report" | grep -qi "integration" || {
        print_error "the drift failure must name the drifted tier, got: $drift_report"
        exit 1
    }

    printf '%s\n' '[tier-pyramid-contract] case=update-baseline-absorbs-the-drift status=start'
    "$pyramid_script" --repository-root "$fixture_root" --update-baseline > /dev/null 2>&1
    "$pyramid_script" --repository-root "$fixture_root" > /dev/null 2>&1 || {
        print_error "after an explicit baseline update the distribution must pass again"
        exit 1
    }

    printf '%s\n' '[tier-pyramid-contract] status=success'
}

main "$@"
