use astronomical_experimental_aligned_expert_packs::AlignedExpertPackPreparer;

#[test]
#[ignore = "inspects an installed research model without creating aligned expert packs"]
fn should_inspect_the_downloaded_model_before_explicit_pack_preparation() {
    let model_directory = super::configured_model_directory_by_id("Ornith-1.0-35B-8bit")
        .expect("the uniform Ornith 8-bit checkpoint should be installed");
    let preparer = AlignedExpertPackPreparer::for_model_directory(&model_directory)
        .expect("the downloaded model should support aligned expert-pack preparation");

    let preparation_inspection = preparer
        .inspect()
        .expect("aligned expert-pack preparation should inspect without mutation");

    assert_eq!(preparation_inspection.total_layer_count, 40);
    assert!(preparation_inspection.total_pack_byte_count > 0);
    assert!(preparation_inspection.available_byte_count > 0);
}
