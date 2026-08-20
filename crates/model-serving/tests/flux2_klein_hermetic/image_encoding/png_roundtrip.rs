use astronomical_model_serving::Flux2KleinPngEncoder;
use image::ImageFormat;

#[test]
fn should_encode_deterministic_lossless_png_and_round_trip_every_generated_pixel() {
    let decoded_rgb = vec![
        -1.0,
        -0.992_156_86,
        -0.984_313_7,
        -0.003_921_569,
        0.0,
        0.003_921_569,
        0.984_313_7,
        0.992_156_86,
        1.0,
        -0.929_411_77,
        -0.850_980_4,
        -0.772_549,
    ];
    let expected_rgb = vec![0, 1, 2, 127, 128, 128, 253, 254, 255, 9, 19, 29];
    let first_png = Flux2KleinPngEncoder::encode_decoded_rgb(2, 2, &decoded_rgb)
        .expect("valid decoded RGB image");
    let second_png = Flux2KleinPngEncoder::encode_decoded_rgb(2, 2, &decoded_rgb)
        .expect("valid decoded RGB image");

    assert_eq!(first_png, second_png);
    let decoded = image::load_from_memory_with_format(&first_png, ImageFormat::Png)
        .expect("encoded PNG must decode")
        .into_rgb8();
    assert_eq!(decoded.dimensions(), (2, 2));
    assert_eq!(decoded.into_raw(), expected_rgb);
}

#[test]
fn should_reject_rgb_payloads_that_do_not_match_the_declared_dimensions() {
    assert!(Flux2KleinPngEncoder::encode_rgb8(2, 2, &[0; 11]).is_err());
    assert!(Flux2KleinPngEncoder::encode_rgb8(0, 2, &[]).is_err());
}
