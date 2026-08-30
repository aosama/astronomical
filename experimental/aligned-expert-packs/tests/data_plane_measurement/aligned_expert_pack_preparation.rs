use astronomical_experimental_aligned_expert_packs::AlignedExpertPackPreparer;

#[test]
#[ignore = "inspects an installed research model without creating aligned expert packs"]
fn should_inspect_the_downloaded_model_before_explicit_pack_preparation() {
    let model_directory =
        super::configured_model_directory_by_id(super::large_sparse_moe_model_id())
            .expect("the large sparse MoE e2e fixture should be installed");
    let preparer = AlignedExpertPackPreparer::for_model_directory(&model_directory)
        .expect("the downloaded model should support aligned expert-pack preparation");

    let preparation_inspection = preparer
        .inspect()
        .expect("aligned expert-pack preparation should inspect without mutation");

    // Layer count must be positive and match the model's own config.
    assert!(
        preparation_inspection.total_layer_count > 0,
        "layer count must be positive"
    );
    assert!(preparation_inspection.total_pack_byte_count > 0);
    assert!(preparation_inspection.available_byte_count > 0);
}
