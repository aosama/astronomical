use std::collections::BTreeSet;

use super::narrowed_alias::{alias_text_from_config, narrowed_alias_arguments};
use super::workspace_graph_tests::fictional_graph;

const INTEGRATION_ALIAS: &str =
    "test --no-fail-fast -p core -p engine -p daemon -p e2e --lib --test integration_tests";
const REST_ALIAS: &str = "nextest archive --archive-file rest.tar.zst -p core -p engine -p daemon -p e2e --test rest_api_tests";

fn names(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn should_keep_only_selected_packages_and_their_test_targets() {
    let narrowed = narrowed_alias_arguments(
        INTEGRATION_ALIAS,
        &words("--no-run --test e2e"),
        &names(&["engine"]),
        &fictional_graph(),
    );
    assert_eq!(
        narrowed,
        Some(words(
            "test --no-fail-fast -p engine --lib --test integration_tests --no-run"
        ))
    );
}

#[test]
fn should_keep_the_library_tests_when_no_requested_test_target_is_selected() {
    let narrowed = narrowed_alias_arguments(
        INTEGRATION_ALIAS,
        &[],
        &names(&["core"]),
        &fictional_graph(),
    );
    assert_eq!(
        narrowed,
        Some(words("test --no-fail-fast -p core --lib")),
        "core has no integration_tests target"
    );
}

#[test]
fn should_build_nothing_when_only_unselected_test_targets_were_asked_for() {
    assert_eq!(
        narrowed_alias_arguments(REST_ALIAS, &[], &names(&["engine"]), &fictional_graph()),
        None
    );
    assert_eq!(
        narrowed_alias_arguments(REST_ALIAS, &[], &names(&["daemon"]), &fictional_graph()),
        Some(words(
            "nextest archive --archive-file rest.tar.zst -p daemon --test rest_api_tests"
        ))
    );
}

#[test]
fn should_build_nothing_when_no_alias_package_is_selected() {
    assert_eq!(
        narrowed_alias_arguments(
            INTEGRATION_ALIAS,
            &[],
            &names(&["tool"]),
            &fictional_graph()
        ),
        None
    );
    assert_eq!(
        narrowed_alias_arguments(INTEGRATION_ALIAS, &[], &BTreeSet::new(), &fictional_graph()),
        None
    );
}

#[test]
fn should_read_an_alias_only_from_the_alias_table() {
    let config_text = "[env]\ntest-integration = \"wrong\"\n\n[alias]\n# comment\ntest-integration = \"test -p core --lib\"\nother = \"x\"\n";
    assert_eq!(
        alias_text_from_config(config_text, "test-integration").as_deref(),
        Some("test -p core --lib")
    );
    assert_eq!(alias_text_from_config(config_text, "missing"), None);
}
