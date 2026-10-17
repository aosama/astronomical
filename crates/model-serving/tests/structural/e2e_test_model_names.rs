#[test]
fn should_load_a_discovered_leaf_model_id_for_every_e2e_test_role() {
    let e2e_test_model_ids = [
        astronomical_test_harness::fixtures::large_sparse_moe_model_id(),
        astronomical_test_harness::fixtures::resident_sparse_moe_model_id(),
        astronomical_test_harness::fixtures::small_dense_model_id(),
        astronomical_test_harness::fixtures::flux2_klein_model_id(),
        astronomical_test_harness::fixtures::k2_horizon_mova_model_id(),
    ];
    assert_eq!(
        e2e_test_model_ids,
        astronomical_test_harness::fixtures::e2e_test_model_ids()
    );
    assert!(
        e2e_test_model_ids
            .iter()
            .all(|model_id| !model_id.is_empty())
    );
}
