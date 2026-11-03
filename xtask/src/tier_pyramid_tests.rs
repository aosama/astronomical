//! The tier pyramid on a fixture tree: each tier is counted from the tree,
//! a recorded baseline passes within tolerance, growth beyond it fails, and
//! an intended migration is absorbed by re-recording the baseline. Typed
//! port of the retired `test-report-tier-pyramid-contract.sh`.

use std::path::Path;

use super::tier_pyramid::{count_tiers, run};

fn write_fixture_file(file_path: &Path, file_text: &str) {
    std::fs::create_dir_all(file_path.parent().expect("a parent")).expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
}

/// Two unit tests beside source, two integration tests plus a support
/// helper that must not count, one e2e journey, and one real-model journey.
fn write_tier_fixture(fixture_root: &Path) {
    write_fixture_file(
        &fixture_root.join("crates/example/src/example_tests.rs"),
        "#[test]\nfn first_unit_case() {\n    assert_eq!(1 + 1, 2);\n}\n\n#[tokio::test]\nasync fn second_unit_case() {\n    assert_eq!(2 + 2, 4);\n}\n",
    );
    write_fixture_file(
        &fixture_root.join("crates/example/tests/integration/journeys.rs"),
        "#[test]\nfn first_integration_case() {}\n\n#[test]\nfn second_integration_case() {}\n",
    );
    write_fixture_file(
        &fixture_root.join("crates/example/tests/support/helper.rs"),
        "#[test]\nfn helper_must_not_count_as_a_test() {}\n",
    );
    write_fixture_file(
        &fixture_root.join("verification/e2e/tests/product_journey.rs"),
        "#[tokio::test]\nasync fn one_end_to_end_case() {}\n",
    );
    write_fixture_file(
        &fixture_root.join("verification/e2e/tests/real_model_journey.rs"),
        "// ModelFidelity::current() selects the explicit request's model fidelity.\n#[tokio::test]\nasync fn one_acceptance_case() {}\n",
    );
}

#[test]
fn should_count_each_tier_from_the_tree_and_exclude_support_helpers() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    write_tier_fixture(fixture_directory.path());

    assert_eq!(count_tiers(fixture_directory.path()), [2, 2, 1, 1]);
}

#[test]
fn should_guide_rather_than_fail_without_a_baseline() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    write_tier_fixture(fixture_directory.path());

    assert_eq!(run(fixture_directory.path(), false), Ok(()));
}

#[test]
fn should_record_a_baseline_then_fail_on_drift_until_it_is_re_recorded() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    write_tier_fixture(fixture_directory.path());
    run(fixture_directory.path(), true).expect("recording the baseline");
    assert!(
        fixture_directory
            .path()
            .join("scripts/ci/tier-pyramid-baseline.json")
            .is_file()
    );
    assert_eq!(
        run(fixture_directory.path(), false),
        Ok(()),
        "no tier drifted"
    );

    write_fixture_file(
        &fixture_directory
            .path()
            .join("crates/example/tests/integration/more_journeys.rs"),
        "#[test]\nfn third_integration_case() {}\n#[test]\nfn fourth_integration_case() {}\n",
    );
    assert!(
        run(fixture_directory.path(), false).is_err(),
        "integration grew beyond the tolerance"
    );

    run(fixture_directory.path(), true).expect("re-recording absorbs the intended migration");
    assert_eq!(run(fixture_directory.path(), false), Ok(()));
}
