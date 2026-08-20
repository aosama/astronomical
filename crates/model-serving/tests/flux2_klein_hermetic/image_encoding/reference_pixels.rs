use astronomical_model_serving::flux2_klein_reference_rgb_u8;

#[test]
fn should_match_the_exact_reference_clamp_scale_and_ties_to_even_conversion() {
    let rgb = flux2_klein_reference_rgb_u8(&[
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        -126.5 / 127.5,
        -125.5 / 127.5,
    ])
    .expect("finite decoded pixels");

    assert_eq!(rgb, vec![0, 0, 64, 128, 191, 255, 255, 1, 2]);
}

#[test]
fn should_reject_non_finite_decoded_pixels_instead_of_hiding_corruption() {
    assert!(flux2_klein_reference_rgb_u8(&[f32::NAN, 0.0, 0.0]).is_err());
    assert!(flux2_klein_reference_rgb_u8(&[f32::INFINITY, 0.0, 0.0]).is_err());
}
