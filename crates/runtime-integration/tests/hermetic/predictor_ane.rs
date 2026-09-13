//! Core ML predictor loader fails open when the compiled program is missing.

#[cfg(target_os = "macos")]
#[test]
fn missing_core_ml_program_does_not_load() {
    assert!(
        astronomical_runtime_integration::PredictorAneEngine::try_load(std::path::Path::new(
            "/this-predictor-program-does-not-exist.mlpackage"
        ))
        .is_none()
    );
}
