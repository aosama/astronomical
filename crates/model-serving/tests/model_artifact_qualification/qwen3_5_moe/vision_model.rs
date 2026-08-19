use std::io::Cursor;

use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5VisionModel};
use astronomical_runtime_integration::{MlxDtype, MlxRuntime};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

#[tokio::test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
async fn should_project_one_minimum_sized_image_into_text_embedding_width() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let mut validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Ornith artifact should validate before vision loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let runtime = MlxRuntime::initialize(mlx_memory_limits)
        .expect("the direct MLX runtime should initialize");
    let vision_model = Qwen3_5VisionModel::load_from_sidecar(&runtime, &mut validated_artifact)
        .expect("the vision sidecar should load")
        .expect("the Ornith artifact should include a vision sidecar");
    let encoded_image_bytes = one_pixel_png();
    let processed_image = crate::common::qwen3_5_moe::certified_ornith_image_processor()
        .process_image_bytes(&encoded_image_bytes)
        .expect("the one-pixel image should expand to the minimum supported grid");
    let expected_visual_token_count =
        i32::try_from(processed_image.image_token_count_after_spatial_merge)
            .expect("the processed visual token count should fit an MLX shape dimension");
    let expected_text_embedding_width = i32::try_from(validated_artifact.config().hidden_size())
        .expect("the configured hidden size should fit an MLX shape dimension");

    let visual_embeddings = vision_model
        .forward(&runtime, &[processed_image])
        .expect("the minimum image should execute through the vision tower");
    runtime
        .evaluate_arrays(&[&visual_embeddings])
        .expect("the visual embeddings should evaluate on the GPU");

    let embedding_shape = visual_embeddings.shape();
    assert_eq!(
        embedding_shape,
        vec![expected_visual_token_count, expected_text_embedding_width],
        "visual embeddings must align processed image tokens with the configured text width"
    );

    let float_visual_embeddings = runtime
        .astype(&visual_embeddings, MlxDtype::Float32)
        .expect("the visual embeddings should cast to float32 for parity inspection");
    let first_visual_embedding_values = runtime
        .slice(&float_visual_embeddings, &[0, 0], &[1, 8], &[1, 1])
        .expect("the first visual embedding prefix should be sliceable")
        .to_vec_f32()
        .expect("the first visual embedding prefix should copy to Rust");
    for (index, &value) in first_visual_embedding_values.iter().enumerate() {
        assert!(
            value.is_finite(),
            "visual embedding component at index {index} must be finite, got {value}"
        );
    }
    let embedding_sums_by_dimension = runtime
        .sum_axis(&float_visual_embeddings, 0, false)
        .expect("visual embeddings should sum across image tokens");
    let complete_embedding_sum = runtime
        .sum_axis(&embedding_sums_by_dimension, 0, false)
        .expect("visual embeddings should sum across hidden dimensions")
        .to_vec_f32()
        .expect("the complete embedding sum should copy to Rust");
    // A nonzero finite reduction catches invalid or disconnected execution
    // without coupling qualification to one packaging variant's float values.
    assert!(
        complete_embedding_sum[0].is_finite() && complete_embedding_sum[0] != 0.0,
        "the complete embedding sum must be a non-zero finite value, got {}",
        complete_embedding_sum[0]
    );
}

fn one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the in-memory one-pixel PNG should encode");
    encoded_image_bytes.into_inner()
}
