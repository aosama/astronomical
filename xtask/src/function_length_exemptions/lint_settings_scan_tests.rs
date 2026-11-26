use super::{
    is_cargo_manifest_file_name, is_clippy_configuration_file_name, manifest_exemptions,
    root_clippy_configuration_exemptions,
};
use crate::function_length_exemptions::{ExemptionKind, FunctionLengthExemption};

fn exemption_kinds(exemptions: &[FunctionLengthExemption]) -> Vec<ExemptionKind> {
    let mut kinds = Vec::with_capacity(exemptions.len());
    for exemption in exemptions {
        kinds.push(exemption.kind);
    }
    kinds
}

#[test]
fn should_accept_a_package_that_inherits_the_workspace_lints() {
    let manifest_text = "[package]\nname = \"thing\"\n\n[lints]\nworkspace = true\n";

    assert!(manifest_exemptions("crates/thing/Cargo.toml", manifest_text).is_empty());
}

#[test]
fn should_accept_a_package_whose_own_table_denies_the_lint() {
    let manifest_text = concat!(
        "[package]\nname = \"thing\"\n\n[lints.clippy]\n",
        "all = { level = \"warn\", priority = -1 }\n",
        "too_many_lines = \"deny\"\n",
    );

    assert!(manifest_exemptions("crates/thing/Cargo.toml", manifest_text).is_empty());
}

#[test]
fn should_report_a_package_whose_own_table_leaves_the_lint_out() {
    let manifest_text = concat!(
        "[package]\nname = \"thing\"\n\n[lints.clippy]\n",
        "all = { level = \"warn\", priority = -1 }\n",
    );

    let exemptions = manifest_exemptions("crates/thing/Cargo.toml", manifest_text);

    assert_eq!(
        exemption_kinds(&exemptions),
        [ExemptionKind::ManifestWithoutDeny]
    );
}

#[test]
fn should_report_the_line_that_sets_the_lint_below_deny() {
    let manifest_text = concat!(
        "[package]\nname = \"thing\"\n\n[lints.clippy]\n",
        "too_many_lines = { level = \"warn\", priority = 1 }\n",
    );

    let exemptions = manifest_exemptions("crates/thing/Cargo.toml", manifest_text);

    assert_eq!(
        exemption_kinds(&exemptions),
        [
            ExemptionKind::ManifestLintLevel,
            ExemptionKind::ManifestWithoutDeny
        ]
    );
    assert_eq!(exemptions[0].line_number, 5);
}

#[test]
fn should_require_the_workspace_root_to_deny_the_lint() {
    let denying_root =
        "[workspace]\nmembers = []\n\n[workspace.lints.clippy]\ntoo_many_lines = \"deny\"\n";
    let silent_root = "[workspace]\nmembers = []\n\n[workspace.lints.clippy]\nall = \"warn\"\n";

    assert!(manifest_exemptions("Cargo.toml", denying_root).is_empty());
    assert_eq!(
        exemption_kinds(&manifest_exemptions("Cargo.toml", silent_root)),
        [ExemptionKind::ManifestWithoutDeny]
    );
}

#[test]
fn should_ignore_the_lint_name_outside_the_clippy_lints_tables() {
    let manifest_text = concat!(
        "[package]\nname = \"thing\"\n\n[lints]\nworkspace = true\n\n",
        "[[bin]]\nname = \"too_many_lines\"\n\n",
        "[package.metadata.notes]\ntoo_many_lines = \"allow\"\n",
    );

    assert!(manifest_exemptions("crates/thing/Cargo.toml", manifest_text).is_empty());
}

#[test]
fn should_accept_a_root_clippy_configuration_set_to_the_budget() {
    let configuration_text = "# Function-length budget\ntoo-many-lines-threshold = 80\n";

    assert!(root_clippy_configuration_exemptions(Some(configuration_text)).is_empty());
}

#[test]
fn should_report_the_threshold_line_when_it_differs_from_the_budget() {
    let configuration_text = "# Function-length budget\ntoo-many-lines-threshold = 100\n";

    let exemptions = root_clippy_configuration_exemptions(Some(configuration_text));

    assert_eq!(
        exemption_kinds(&exemptions),
        [ExemptionKind::BudgetThresholdMismatch]
    );
    assert_eq!(exemptions[0].line_number, 2);
}

#[test]
fn should_report_a_root_clippy_configuration_without_a_threshold_or_missing() {
    let without_threshold = root_clippy_configuration_exemptions(Some("msrv = \"1.85\"\n"));
    let missing = root_clippy_configuration_exemptions(None);

    assert_eq!(
        exemption_kinds(&without_threshold),
        [ExemptionKind::BudgetThresholdMismatch]
    );
    assert_eq!(
        exemption_kinds(&missing),
        [ExemptionKind::BudgetThresholdMismatch]
    );
}

#[test]
fn should_recognize_manifests_and_both_clippy_configuration_names() {
    assert!(is_cargo_manifest_file_name("Cargo.toml"));
    assert!(!is_cargo_manifest_file_name("Cargo.lock"));
    assert!(is_clippy_configuration_file_name("clippy.toml"));
    assert!(is_clippy_configuration_file_name(".clippy.toml"));
    assert!(!is_clippy_configuration_file_name("rustfmt.toml"));
}
