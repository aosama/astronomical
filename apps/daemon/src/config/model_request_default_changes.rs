//! Separates a model's request defaults from the rest of its policy when a
//! configuration reload compares two resolved catalogs.
//!
//! Temperature, top p and repetition penalty are filled into each request by
//! the daemon as the request arrives (`request_generation_defaults`). The
//! worker never holds them, so changing one needs no worker replacement.
//! Treating them like the rest of a model's policy made every such change
//! wait for a pause in generation, and a coding agent that sends one request
//! after another never pauses: the saved value stayed unapplied until the
//! engine restarted.
//!
//! The maximum output tokens also sits in the request defaults, but the
//! worker's model configuration carries its own copy, so a change to it still
//! differs outside the request defaults and still replaces the worker.

use std::collections::HashMap;

use crate::RuntimeModelPolicy;

type ModelPolicyCatalog = HashMap<String, RuntimeModelPolicy>;

/// True when the catalogs differ in anything a replacement worker must pick
/// up: a model added or removed, or any policy field other than the request
/// defaults.
pub(crate) fn model_policies_differ_beyond_request_defaults(
    current_catalog: &ModelPolicyCatalog,
    candidate_catalog: &ModelPolicyCatalog,
) -> bool {
    if current_catalog.len() != candidate_catalog.len() {
        return true;
    }
    for (model_id, current_policy) in current_catalog {
        let Some(candidate_policy) = candidate_catalog.get(model_id) else {
            return true;
        };
        let mut candidate_policy_with_current_defaults = candidate_policy.clone();
        candidate_policy_with_current_defaults.generation_defaults =
            current_policy.generation_defaults;
        if candidate_policy_with_current_defaults != *current_policy {
            return true;
        }
    }
    false
}

/// True when any model present in both catalogs has different request defaults.
pub(crate) fn model_request_defaults_differ(
    current_catalog: &ModelPolicyCatalog,
    candidate_catalog: &ModelPolicyCatalog,
) -> bool {
    for (model_id, current_policy) in current_catalog {
        if let Some(candidate_policy) = candidate_catalog.get(model_id)
            && candidate_policy.generation_defaults != current_policy.generation_defaults
        {
            return true;
        }
    }
    false
}

/// The live catalog with each model's request defaults taken from the
/// candidate. Used when the rest of a reload waits for an application
/// restart: the request defaults are safe to apply at once, while the
/// directories and worker settings stay as the running worker knows them.
pub(crate) fn catalog_with_candidate_request_defaults(
    live_catalog: &ModelPolicyCatalog,
    candidate_catalog: &ModelPolicyCatalog,
) -> ModelPolicyCatalog {
    let mut updated_catalog = live_catalog.clone();
    for (model_id, live_policy) in &mut updated_catalog {
        if let Some(candidate_policy) = candidate_catalog.get(model_id) {
            live_policy.generation_defaults = candidate_policy.generation_defaults;
        }
    }
    updated_catalog
}
