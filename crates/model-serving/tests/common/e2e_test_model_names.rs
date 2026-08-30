//! Leaf model identities used by ignored end-to-end tests.
//!
//! Artifact names change as we retarget checkpoints. Tests look up a stable role
//! from `registry/e2e_test_model_names.json` instead of embedding those names.

use std::collections::BTreeMap;
use std::sync::OnceLock;

const E2E_TEST_MODEL_NAMES_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../registry/e2e_test_model_names.json"
));
const LARGE_SPARSE_MOE_ROLE: &str = "large_sparse_moe";
const RESIDENT_SPARSE_MOE_ROLE: &str = "resident_sparse_moe";
const LAGUNA_XS_ROLE: &str = "laguna_xs";
const DENSE_MTP_ROLE: &str = "dense_mtp";
const SMALL_DENSE_ROLE: &str = "small_dense";
const FLUX2_KLEIN_ROLE: &str = "flux2_klein";
const E2E_TEST_MODEL_ROLES: [&str; 6] = [
    LARGE_SPARSE_MOE_ROLE,
    RESIDENT_SPARSE_MOE_ROLE,
    LAGUNA_XS_ROLE,
    DENSE_MTP_ROLE,
    SMALL_DENSE_ROLE,
    FLUX2_KLEIN_ROLE,
];

fn e2e_test_model_names() -> &'static BTreeMap<String, String> {
    static NAMES: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let names: BTreeMap<String, String> = serde_json::from_str(E2E_TEST_MODEL_NAMES_JSON)
            .expect(
            "registry/e2e_test_model_names.json should map each e2e role to a discovered leaf model id",
        );
        for role in E2E_TEST_MODEL_ROLES {
            let model_id = names.get(role).unwrap_or_else(|| {
                panic!("registry/e2e_test_model_names.json must declare {role}")
            });
            assert_leaf_model_id(role, model_id);
        }
        assert_eq!(
            names.len(),
            E2E_TEST_MODEL_ROLES.len(),
            "registry/e2e_test_model_names.json must declare only the known e2e roles"
        );
        names
    })
}

fn assert_leaf_model_id(role: &str, model_id: &str) {
    assert!(
        !model_id.is_empty(),
        "registry/e2e_test_model_names.json role {role} must declare a leaf model id"
    );
    assert_eq!(
        model_id,
        model_id.trim(),
        "registry/e2e_test_model_names.json role {role} must not pad the leaf model id"
    );
    assert!(
        !model_id.contains('/') && !model_id.contains('\\'),
        "registry/e2e_test_model_names.json role {role} must be a discovered leaf id, not a path or provider id"
    );
}

fn model_id_for_role(role: &str) -> &'static str {
    e2e_test_model_names()
        .get(role)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("registry/e2e_test_model_names.json must declare {role}"))
}

pub(crate) fn large_sparse_moe_model_id() -> &'static str {
    model_id_for_role(LARGE_SPARSE_MOE_ROLE)
}

pub(crate) fn resident_sparse_moe_model_id() -> &'static str {
    model_id_for_role(RESIDENT_SPARSE_MOE_ROLE)
}

pub(crate) fn laguna_xs_model_id() -> &'static str {
    model_id_for_role(LAGUNA_XS_ROLE)
}

pub(crate) fn dense_mtp_model_id() -> &'static str {
    model_id_for_role(DENSE_MTP_ROLE)
}

pub(crate) fn small_dense_model_id() -> &'static str {
    model_id_for_role(SMALL_DENSE_ROLE)
}

pub(crate) fn flux2_klein_model_id() -> &'static str {
    model_id_for_role(FLUX2_KLEIN_ROLE)
}

pub(crate) fn e2e_test_model_ids() -> [&'static str; 6] {
    [
        large_sparse_moe_model_id(),
        resident_sparse_moe_model_id(),
        laguna_xs_model_id(),
        dense_mtp_model_id(),
        small_dense_model_id(),
        flux2_klein_model_id(),
    ]
}

/// Roles remaining chat/Laguna journeys require on disk. FLUX is optional here
/// because its own tests fail closed at discovery.
pub(crate) fn required_e2e_test_model_ids() -> [&'static str; 5] {
    [
        large_sparse_moe_model_id(),
        resident_sparse_moe_model_id(),
        laguna_xs_model_id(),
        dense_mtp_model_id(),
        small_dense_model_id(),
    ]
}
