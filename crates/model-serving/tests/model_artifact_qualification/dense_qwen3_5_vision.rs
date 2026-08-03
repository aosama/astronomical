use std::io::Cursor;
use std::path::PathBuf;
use std::time::Duration;

use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5MoEImageProcessor, Qwen3_5Model,
};
use astronomical_runtime_integration::MlxRuntime;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use tokio::time::timeout;

const DENSE_QWEN3_5_VISION_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_DENSE_QWEN3_5_VISION_MODEL_DIRECTORY";

#[tokio::test]
#[ignore = "requires ASTRONOMICAL_DENSE_QWEN3_5_VISION_MODEL_DIRECTORY"]
async fn should_load_a_dense_qwen3_5_vision_model_and_project_a_minimum_image() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let Some(model_directory) = configured_dense_qwen3_5_vision_model_directory() else {
        eprintln!("[dense-qwen3-5-vision] status=skipped reason=required_model_directory_missing");
        return;
    };
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

    timeout(Duration::from_secs(115), async move {
        tokio::task::spawn_blocking(move || {
            eprintln!("[dense-qwen3-5-vision] status=progress phase=artifact_validation");
            let validated_artifact = Qwen3_5ArtifactValidator::new()
                .validate(&model_directory, 20_480)
                .expect("the dense Qwen3.5 vision artifact should validate");
            assert!(validated_artifact.config().is_dense_model());
            assert!(validated_artifact.supports_image_input());
            let vision_config = validated_artifact
                .vision_config()
                .expect("the validated vision model should retain vision_config");
            assert_eq!(
                vision_config.out_hidden_size(),
                validated_artifact.config().hidden_size(),
                "the vision projection width should match the text hidden width"
            );
            let expected_visual_embedding_width = i32::try_from(vision_config.out_hidden_size())
                .expect("the vision projection width should fit the MLX shape type");
            let image_processor = Qwen3_5MoEImageProcessor::from_vision_config(vision_config);

            eprintln!("[dense-qwen3-5-vision] status=progress phase=runtime_initialization");
            let runtime =
                MlxRuntime::initialize(mlx_memory_limits).expect("MLX runtime should initialize");

            eprintln!("[dense-qwen3-5-vision] status=progress phase=model_loading");
            let dense_qwen3_5_vision_model =
                Qwen3_5Model::load(runtime, validated_artifact, &model_directory, false)
                    .expect("the dense Qwen3.5 vision model should load");
            let vision_model = dense_qwen3_5_vision_model
                .vision_model()
                .expect("the dense Qwen3.5 model should bind its vision tower");

            eprintln!("[dense-qwen3-5-vision] status=progress phase=vision_projection");
            let processed_image = image_processor
                .process_image_bytes(&one_pixel_png())
                .expect("the one-pixel PNG should preprocess");
            let visual_embeddings = vision_model
                .forward(dense_qwen3_5_vision_model.runtime(), &[processed_image])
                .expect("the vision tower should project the image");
            dense_qwen3_5_vision_model
                .runtime()
                .evaluate_arrays(&[&visual_embeddings])
                .expect("the visual embeddings should evaluate");
            let visual_embedding_shape = visual_embeddings.shape();
            assert_eq!(visual_embedding_shape.len(), 2);
            assert!(visual_embedding_shape[0] > 0);
            assert_eq!(visual_embedding_shape[1], expected_visual_embedding_width);
            eprintln!("[dense-qwen3-5-vision] status=complete");
        })
        .await
        .expect("the dense Qwen3.5 vision task should not panic");
    })
    .await
    .expect("the dense Qwen3.5 vision smoke test must finish within 115 seconds");
}

fn configured_dense_qwen3_5_vision_model_directory() -> Option<PathBuf> {
    std::env::var_os(DENSE_QWEN3_5_VISION_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE).map(PathBuf::from)
}

fn one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the in-memory PNG should encode");
    encoded_image_bytes.into_inner()
}
