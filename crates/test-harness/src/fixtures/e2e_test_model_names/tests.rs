use super::*;

#[test]
fn should_load_exactly_the_configured_roles_as_leaf_model_ids() {
    let model_id_by_role = e2e_test_model_names();
    let mut expected_roles = E2E_TEST_MODEL_ROLES.to_vec();
    expected_roles.sort_unstable();
    let registry_roles = model_id_by_role
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();

    assert_eq!(registry_roles, expected_roles);
    for (role, model_id) in model_id_by_role {
        assert_leaf_model_id(role, model_id);
    }
}

#[test]
fn should_return_registry_model_ids_in_the_declared_role_order() {
    let model_id_by_role = e2e_test_model_names();
    let expected_model_ids = E2E_TEST_MODEL_ROLES.map(|role| {
        model_id_by_role
            .get(role)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("registry/e2e_test_model_names.json must declare {role}"))
    });

    assert_eq!(large_sparse_moe_streaming_model_id(), expected_model_ids[0]);
    assert_eq!(resident_sparse_moe_model_id(), expected_model_ids[1]);
    assert_eq!(ornith_dflash_draft_model_id(), expected_model_ids[2]);
    assert_eq!(small_dense_model_id(), expected_model_ids[3]);
    assert_eq!(flux2_klein_model_id(), expected_model_ids[4]);
    assert_eq!(e2e_test_model_ids(), expected_model_ids);
    assert_eq!(
        required_e2e_test_model_ids(),
        [
            expected_model_ids[0],
            expected_model_ids[1],
            expected_model_ids[2],
        ]
    );
}

#[test]
#[should_panic(expected = "must declare a leaf model id")]
fn should_reject_an_empty_model_id() {
    assert_leaf_model_id("test_role", "");
}

#[test]
#[should_panic(expected = "must not pad the leaf model id")]
fn should_reject_a_whitespace_padded_model_id() {
    assert_leaf_model_id("test_role", " model-id ");
}

#[test]
#[should_panic(expected = "not a path or provider id")]
fn should_reject_a_slash_separated_model_id() {
    assert_leaf_model_id("test_role", "provider/model-id");
}

#[test]
#[should_panic(expected = "not a path or provider id")]
fn should_reject_a_backslash_separated_model_id() {
    assert_leaf_model_id("test_role", "provider\\model-id");
}
