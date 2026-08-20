use astronomical_model_serving::{
    Flux2KleinPackedLatentLayout, flux2_klein_inverse_batch_norm_reference,
};

#[test]
fn should_reverse_packed_latents_into_the_reference_channel_and_patch_order() {
    let layout = Flux2KleinPackedLatentLayout::new(1, 2, 3).expect("valid packed layout");

    assert_eq!(layout.packed_shape(), [1, 6, 128]);
    assert_eq!(layout.unpatchified_shape(), [1, 4, 6, 32]);
    assert_eq!(layout.unpatchified_source(0, 0, 0, 0), Some([0, 0, 0]));
    assert_eq!(layout.unpatchified_source(0, 1, 1, 0), Some([0, 0, 3]));
    assert_eq!(layout.unpatchified_source(0, 2, 4, 7), Some([0, 5, 28]));
    assert_eq!(layout.unpatchified_source(0, 3, 5, 31), Some([0, 5, 127]));
}

#[test]
fn should_reject_a_packed_sequence_that_does_not_match_the_image_geometry() {
    let layout = Flux2KleinPackedLatentLayout::new(1, 2, 3).expect("valid packed layout");

    assert!(layout.validate_packed_shape(&[1, 5, 128]).is_err());
    assert!(layout.validate_packed_shape(&[1, 6, 127]).is_err());
}

#[test]
fn should_apply_inverse_batch_norm_with_running_statistics_and_reference_epsilon() {
    let restored = flux2_klein_inverse_batch_norm_reference(
        &[0.0, 1.0, -1.0, 2.0],
        2,
        &[10.0, -3.0],
        &[3.999_9, 8.999_9],
    )
    .expect("valid batch-normalization geometry");

    assert_eq!(restored, vec![10.0, 0.0, 8.0, 3.0]);
}
