//! The channel isolation scan on fixture trees: Development state passes,
//! every Stable reference form fails, the config loader's own tests are
//! exempt, and a new package test tree is scanned without configuration.
//! Typed port of the retired `test-channel-isolation-checker-contract.sh`.

use std::path::Path;

use super::test_channel_isolation::{
    channel_isolation_violations, line_reaches_for_stable_state, run, scanned_test_trees,
};

fn write_fixture_file(file_path: &Path, file_text: &str) {
    std::fs::create_dir_all(file_path.parent().expect("a parent")).expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
}

#[test]
fn should_accept_development_state_references() {
    for development_line in [
        "let state = home.join(\".astronomical-dev\");",
        "Config::load_development(&temporary_home)",
        "AstronomicalRuntimeInstance::Development",
        "// the Stable app is never touched here",
    ] {
        assert!(
            !line_reaches_for_stable_state(development_line),
            "{development_line}"
        );
    }
}

#[test]
fn should_reject_every_form_of_stable_state_reference() {
    for stable_line in [
        "Config::load_from_home_directory()",
        "let instance = AstronomicalRuntimeInstance::Stable;",
        "let path = \"~/.astronomical/config.json\";",
        "let label = \"~/.astronomical\";",
        "home.join(\".astronomical\")",
        "home.join(\".astronomical/models\")",
    ] {
        assert!(line_reaches_for_stable_state(stable_line), "{stable_line}");
    }
}

#[test]
fn should_scan_every_package_test_tree_including_verification_but_not_the_config_loader() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let fixture_root = fixture_directory.path();
    for test_file in [
        "apps/daemon/tests/journey.rs",
        "crates/new-crate/tests/contract.rs",
        "verification/e2e/tests/product.rs",
        "crates/config/tests/loader.rs",
    ] {
        write_fixture_file(&fixture_root.join(test_file), "fn placeholder() {}\n");
    }

    let scanned_trees = scanned_test_trees(fixture_root);

    let mut relative_trees = Vec::new();
    for scanned_tree in &scanned_trees {
        relative_trees.push(
            scanned_tree
                .strip_prefix(fixture_root)
                .expect("inside")
                .display()
                .to_string(),
        );
    }
    assert_eq!(
        relative_trees,
        vec![
            "apps/daemon/tests",
            "crates/new-crate/tests",
            "verification/e2e/tests"
        ]
    );
}

#[test]
fn should_report_the_file_and_line_of_each_violation_and_fail_the_command() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let fixture_root = fixture_directory.path();
    write_fixture_file(
        &fixture_root.join("crates/clean/tests/clean.rs"),
        "let state = \".astronomical-dev\";\n",
    );
    write_fixture_file(
        &fixture_root.join("verification/acceptance/tests/stable.rs"),
        "fn setup() {}\nlet config = Config::load_from_home_directory();\n",
    );
    write_fixture_file(
        &fixture_root.join("crates/config/tests/loader.rs"),
        "AstronomicalRuntimeInstance::Stable\n",
    );

    let violations = channel_isolation_violations(fixture_root).expect("the fixture is scannable");

    assert_eq!(
        violations,
        vec![
            "verification/acceptance/tests/stable.rs:2:let config = Config::load_from_home_directory();"
        ]
    );
    assert!(run(fixture_root, &[]).is_err());
    assert!(
        run(fixture_root, &["--fix".to_owned()]).is_err(),
        "the command takes no arguments"
    );
}

#[test]
fn should_pass_on_this_repository() {
    let repository_root = crate::workspace_root_from_manifest_directory();

    assert_eq!(
        channel_isolation_violations(&repository_root),
        Ok(Vec::new())
    );
}
