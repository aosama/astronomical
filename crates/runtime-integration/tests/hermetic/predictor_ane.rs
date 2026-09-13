//! Core ML predictor loader fails open when the compiled program is missing.

#[cfg(target_os = "macos")]
#[test]
fn missing_core_ml_program_does_not_load() {
    assert!(
        astronomical_runtime_integration::PredictorAneEngine::try_load(std::path::Path::new(
            "/this-predictor-program-does-not-exist.mlpackage"
        ))
        .is_err()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rust_written_grouped_convolution_model_predicts() {
    let snapshot = astronomical_runtime_integration::PredictorAneConvolutionSnapshot {
        layer_count: 2,
        expert_count: 8,
        input_dim: 12,
        hidden_dim: 5,
        conv1_weights: vec![0.01; 2 * 5 * 12],
        conv1_bias: vec![0.0; 2 * 5],
        conv2_weights: vec![0.02; 2 * 8 * 5],
        conv2_bias: vec![0.0; 2 * 8],
    };
    let mlmodel_path = std::env::temp_dir().join("astronomical-predictor-hermetic.mlmodel");
    astronomical_runtime_integration::write_predictor_mlmodel(&snapshot, &mlmodel_path)
        .expect("the NeuralNetwork snapshot should serialize");
    let engine = astronomical_runtime_integration::PredictorAneEngine::try_load(&mlmodel_path)
        .unwrap_or_else(|error| panic!("Core ML load failed: {error}"));
    let head_inputs = vec![0.0_f32; 2 * 12];
    let logits = engine
        .predict(&head_inputs, 2, 12, 8)
        .expect("grouped 1x1 convolution predict must return logits");
    assert_eq!(logits.len(), 16);
}
