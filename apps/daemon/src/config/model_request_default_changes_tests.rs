//! Tests for telling a model's request defaults apart from the rest of its
//! policy when two resolved catalogs are compared.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::RuntimeModelPolicy;
use crate::config::config_reload_tests::sample_runtime_model_policy;
use crate::config::model_request_default_changes::{
    catalog_with_candidate_request_defaults, model_policies_differ_beyond_request_defaults,
    model_request_defaults_differ,
};

#[test]
fn should_see_only_a_request_default_change_when_the_temperature_moves() {
    let current_catalog = catalog_with_policy(sample_runtime_model_policy());
    let mut candidate_policy = sample_runtime_model_policy();
    candidate_policy.generation_defaults.temperature_thousandths = Some(100);
    let candidate_catalog = catalog_with_policy(candidate_policy);

    assert!(model_request_defaults_differ(
        &current_catalog,
        &candidate_catalog
    ));
    assert!(!model_policies_differ_beyond_request_defaults(
        &current_catalog,
        &candidate_catalog
    ));
}

#[test]
fn should_see_no_change_between_identical_catalogs() {
    let current_catalog = catalog_with_policy(sample_runtime_model_policy());
    let candidate_catalog = catalog_with_policy(sample_runtime_model_policy());

    assert!(!model_request_defaults_differ(
        &current_catalog,
        &candidate_catalog
    ));
    assert!(!model_policies_differ_beyond_request_defaults(
        &current_catalog,
        &candidate_catalog
    ));
}

#[test]
fn should_report_a_directory_change_as_more_than_a_request_default() {
    let current_catalog = catalog_with_policy(sample_runtime_model_policy());
    let mut candidate_policy = sample_runtime_model_policy();
    candidate_policy.model_directory = PathBuf::from("/tmp/models/moved");
    let candidate_catalog = catalog_with_policy(candidate_policy);

    assert!(model_policies_differ_beyond_request_defaults(
        &current_catalog,
        &candidate_catalog
    ));
    assert!(!model_request_defaults_differ(
        &current_catalog,
        &candidate_catalog
    ));
}

#[test]
fn should_report_an_added_or_removed_model_as_more_than_a_request_default() {
    let current_catalog = catalog_with_policy(sample_runtime_model_policy());
    let mut candidate_catalog = current_catalog.clone();
    candidate_catalog.insert("second".to_owned(), sample_runtime_model_policy());

    assert!(model_policies_differ_beyond_request_defaults(
        &current_catalog,
        &candidate_catalog
    ));
    assert!(model_policies_differ_beyond_request_defaults(
        &candidate_catalog,
        &current_catalog
    ));
}

/// Same model count, different names: a rename must not pass as unchanged.
#[test]
fn should_report_a_renamed_model_as_more_than_a_request_default() {
    let current_catalog = catalog_with_policy(sample_runtime_model_policy());
    let renamed_catalog = HashMap::from([("renamed".to_owned(), sample_runtime_model_policy())]);

    assert!(model_policies_differ_beyond_request_defaults(
        &current_catalog,
        &renamed_catalog
    ));
}

#[test]
fn should_copy_only_the_request_defaults_into_the_live_catalog() {
    let live_catalog = catalog_with_policy(sample_runtime_model_policy());
    let mut candidate_policy = sample_runtime_model_policy();
    candidate_policy.generation_defaults.temperature_thousandths = Some(100);
    candidate_policy.model_directory = PathBuf::from("/tmp/models/moved");
    let candidate_catalog = catalog_with_policy(candidate_policy);

    let updated_catalog =
        catalog_with_candidate_request_defaults(&live_catalog, &candidate_catalog);

    let updated_policy = &updated_catalog["default"];
    assert_eq!(
        updated_policy.generation_defaults.temperature_thousandths,
        Some(100)
    );
    assert_eq!(
        updated_policy.model_directory,
        PathBuf::from("/tmp/models/default")
    );
}

#[test]
fn should_leave_a_model_missing_from_the_candidate_unchanged() {
    let mut live_policy = sample_runtime_model_policy();
    live_policy.generation_defaults.temperature_thousandths = Some(700);
    let live_catalog = catalog_with_policy(live_policy);

    let updated_catalog = catalog_with_candidate_request_defaults(&live_catalog, &HashMap::new());

    assert_eq!(updated_catalog, live_catalog);
}

fn catalog_with_policy(policy: RuntimeModelPolicy) -> HashMap<String, RuntimeModelPolicy> {
    HashMap::from([("default".to_owned(), policy)])
}
