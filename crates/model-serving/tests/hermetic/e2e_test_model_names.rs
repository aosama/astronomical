#[test]
fn should_load_a_discovered_leaf_model_id_for_every_e2e_test_role() {
    let e2e_test_model_ids = [
        crate::common::large_sparse_moe_model_id(),
        crate::common::resident_sparse_moe_model_id(),
        crate::common::laguna_xs_model_id(),
        crate::common::dense_mtp_model_id(),
        crate::common::small_dense_model_id(),
        crate::common::flux2_klein_model_id(),
    ];
    assert_eq!(e2e_test_model_ids, crate::common::e2e_test_model_ids());
    assert!(
        e2e_test_model_ids
            .iter()
            .all(|model_id| !model_id.is_empty())
    );
}
