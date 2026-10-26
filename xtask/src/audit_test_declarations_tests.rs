//! Unit coverage for the pure test-declaration audit: values in, findings out.

use super::audit_test_declarations::{audit_test_declarations, target_paths_from_manifest};
use std::collections::{BTreeMap, BTreeSet};

fn audit(
    crate_name: &str,
    test_sources: &[(&str, &str)],
    bin_target_paths: &[&str],
) -> Vec<String> {
    let sources: BTreeMap<String, String> = test_sources
        .iter()
        .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
        .collect();
    let bin_paths: BTreeSet<String> = bin_target_paths
        .iter()
        .map(|path| (*path).to_owned())
        .collect();
    audit_test_declarations(crate_name, &sources, &bin_paths)
        .into_iter()
        .map(|finding| finding.describe())
        .collect()
}

#[test]
fn should_flag_a_test_file_that_no_module_declares() {
    // The exact shape this audit exists for: an undeclared sibling of mod.rs
    // that sat uncompiled for its whole life.
    let findings = audit(
        "some-worker",
        &[
            ("tests/integration_tests.rs", "mod integration;\n"),
            ("tests/integration/mod.rs", "mod graceful_shutdown;\n"),
            (
                "tests/integration/graceful_shutdown.rs",
                "#[test] fn should_shutdown() {}\n",
            ),
            (
                "tests/integration/flux_reference_oracle.rs",
                "#[test] fn should_parse() {}\n",
            ),
        ],
        &[],
    );
    assert_eq!(
        findings.len(),
        1,
        "expected exactly one orphan: {findings:?}"
    );
    assert!(findings[0].contains("tests/integration/flux_reference_oracle.rs"));
    assert!(findings[0].contains("some-worker"));
    assert!(findings[0].contains("never compiles"));
}

#[test]
fn should_accept_a_file_declared_by_its_sibling_mod_rs() {
    let findings = audit(
        "some-worker",
        &[
            ("tests/gpu_tests.rs", "mod gpu;\n"),
            ("tests/gpu/mod.rs", "mod wired_limit_claim_probe;\n"),
            (
                "tests/gpu/wired_limit_claim_probe.rs",
                "#[test] fn should_probe() {}\n",
            ),
        ],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_accept_a_module_root_relocated_by_a_path_attribute() {
    // The runtime-integration style: the folder's root lives inside the folder
    // (integration/root.rs) so cargo cannot auto-discover it a second time as
    // a standalone target, and the root declares its siblings plainly.
    let findings = audit(
        "some-worker",
        &[
            (
                "tests/integration_tests.rs",
                "#[path = \"integration/root.rs\"]\nmod integration;\n",
            ),
            (
                "tests/integration/root.rs",
                "mod allocator_cleanup;\nmod metallib_path;\n",
            ),
            (
                "tests/integration/allocator_cleanup.rs",
                "#[test] fn should_cleanup() {}\n",
            ),
            (
                "tests/integration/metallib_path.rs",
                "#[test] fn should_resolve() {}\n",
            ),
        ],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_accept_a_file_declared_through_a_path_attribute_with_interleaved_cfg() {
    // The acceptance style: #[cfg(feature)] sits between #[path] and the mod
    // it qualifies, and the relocated file sits two folders deep.
    let findings = audit(
        "acceptance",
        &[
            (
                "tests/chat_serving.rs",
                "#[cfg(feature = \"serving-acceptance\")]\n#[path = \"chat_serving/mod.rs\"]\nmod serving_acceptance;\n#[cfg(feature = \"serving-acceptance\")]\n#[path = \"chat_serving/chat/small_dense_model.rs\"]\nmod small_dense_model;\n",
            ),
            ("tests/chat_serving/mod.rs", "mod support;\n"),
            (
                "tests/chat_serving/chat/small_dense_model.rs",
                "#[test] fn should_serve() {}\n",
            ),
        ],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_treat_bin_targeted_files_as_reachable_roots() {
    // Fixture binaries compile as cargo [[bin]] targets — they are entry
    // points, not modules, and need no `mod` chain.
    let findings = audit(
        "some-worker",
        &[
            ("tests/integration_tests.rs", "mod integration;\n"),
            ("tests/integration/mod.rs", EMPTY_CONTENT),
            ("tests/fixtures/scripted_worker.rs", "fn main() {}\n"),
        ],
        &["tests/fixtures/scripted_worker.rs"],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_ignore_root_level_test_targets() {
    // Cargo auto-discovers tests/<name>.rs as targets; no declaration needed.
    let findings = audit(
        "some-worker",
        &[("tests/standalone_target.rs", "#[test] fn should_run() {}\n")],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_accept_submodules_of_a_file_module_resolved_in_its_sibling_folder() {
    // The CLI style: launch.rs exists as the module file AND launch/ holds its
    // submodules; `mod launch;` resolves to the file, whose `mod codex;`
    // resolves inside the folder the file owns.
    let findings = audit(
        "astronomical-cli",
        &[
            ("tests/integration_tests.rs", "mod integration;\n"),
            ("tests/integration/mod.rs", "mod launch;\nmod support;\n"),
            ("tests/integration/launch.rs", "mod codex;\nmod opencode;\n"),
            (
                "tests/integration/launch/codex.rs",
                "#[test] fn should_launch() {}\n",
            ),
            (
                "tests/integration/launch/opencode.rs",
                "#[test] fn should_launch() {}\n",
            ),
            ("tests/integration/support.rs", "mod environment;\n"),
            (
                "tests/integration/support/environment.rs",
                "#[test] fn should_env() {}\n",
            ),
        ],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_accept_nested_directories_declared_through_the_mod_rs_chain() {
    let findings = audit(
        "some-worker",
        &[
            ("tests/integration_tests.rs", "mod integration;\n"),
            ("tests/integration/mod.rs", "mod gpu;\n"),
            ("tests/integration/gpu/mod.rs", "mod probe;\n"),
            (
                "tests/integration/gpu/probe.rs",
                "#[test] fn should_probe() {}\n",
            ),
        ],
        &[],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_flag_every_unreachable_file_in_an_undeclared_directory() {
    // A directory no target reaches leaves each of its files orphaned; the
    // audit names every one of them rather than hiding behind the folder.
    let findings = audit(
        "some-worker",
        &[
            ("tests/integration_tests.rs", "mod integration;\n"),
            ("tests/integration/mod.rs", EMPTY_CONTENT),
            ("tests/bogus/one.rs", "#[test] fn should_one() {}\n"),
            ("tests/bogus/two.rs", "#[test] fn should_two() {}\n"),
        ],
        &[],
    );
    assert_eq!(
        findings.len(),
        2,
        "both files are unreachable: {findings:?}"
    );
    assert!(findings.iter().any(|f| f.contains("tests/bogus/one.rs")));
    assert!(findings.iter().any(|f| f.contains("tests/bogus/two.rs")));
}

#[test]
fn should_not_flag_an_empty_tests_tree() {
    let findings = audit("some-worker", &[], &[]);
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_flag_an_attribute_adjacency_broken_by_a_statement() {
    // A #[path] only qualifies the next declaration; an intervening statement
    // consumes nothing and the file the attribute named stays unreachable.
    let findings = audit(
        "some-worker",
        &[
            (
                "tests/integration_tests.rs",
                "#[path = \"integration/orphan.rs\"]\nfn main() {}\nmod integration;\n",
            ),
            ("tests/integration/mod.rs", EMPTY_CONTENT),
            (
                "tests/integration/orphan.rs",
                "#[test] fn should_run() {}\n",
            ),
        ],
        &[],
    );
    assert_eq!(findings.len(), 1, "the orphan stays orphaned: {findings:?}");
    assert!(findings[0].contains("tests/integration/orphan.rs"));
}

#[test]
fn should_treat_explicitly_named_test_targets_as_reachable_roots() {
    // The acceptance suite names deep targets explicitly: the file lives at
    // tests/gpu_contracts/direct_mlx.rs and cargo compiles it as a target.
    let findings = audit(
        "acceptance",
        &[(
            "tests/gpu_contracts/direct_mlx.rs",
            "#[test] fn should_contract() {}\n",
        )],
        &["tests/gpu_contracts/direct_mlx.rs"],
    );
    assert!(findings.is_empty(), "unexpected findings: {findings:?}");
}

#[test]
fn should_extract_target_paths_from_both_bin_and_test_sections() {
    let manifest = "\
[package]
name = \"some-worker\"

[[bin]]
name = \"scripted\"
path = \"tests/fixtures/scripted_worker.rs\"

[[test]]
name = \"direct_mlx\"
path = \"tests/gpu_contracts/direct_mlx.rs\"

[dependencies]
serde = \"1\"
";
    let target_paths = target_paths_from_manifest(manifest);
    assert_eq!(target_paths.len(), 2);
    assert!(target_paths.contains("tests/fixtures/scripted_worker.rs"));
    assert!(target_paths.contains("tests/gpu_contracts/direct_mlx.rs"));
}

const EMPTY_CONTENT: &str = "";
