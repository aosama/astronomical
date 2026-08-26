//! Preflights every exact downloaded-model identity used by local qualification.
//!
//! Expensive journeys should fail before loading MLX when a fixture was removed
//! or renamed in Development configuration. Dynamic selection tests are excluded
//! because their contract intentionally chooses by discovered capabilities.

const REQUIRED_MODEL_FIXTURE_IDS: &[&str] = &[
    crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID,
    "Qwen3.8-27B-MTPLX-4bit",
    "Laguna-XS-2.1-oQ8e",
    "FLUX.2-klein-4B",
];

#[test]
#[ignore = "reads Development model_directories to preflight downloaded qualification fixtures"]
fn should_discover_every_exact_downloaded_model_fixture() {
    let discovered_models = crate::common::configured_discovered_models();
    let mut missing_model_fixture_ids = Vec::new();

    for required_model_fixture_id in REQUIRED_MODEL_FIXTURE_IDS {
        eprintln!("[model-fixture-discovery] status=progress model={required_model_fixture_id}");
        if !discovered_models
            .iter()
            .any(|discovered_model| discovered_model.model_id == *required_model_fixture_id)
        {
            missing_model_fixture_ids.push(*required_model_fixture_id);
        }
    }

    assert!(
        missing_model_fixture_ids.is_empty(),
        "Development model_directories must discover every downloaded qualification fixture; missing={missing_model_fixture_ids:?}"
    );

    eprintln!(
        "[model-fixture-discovery] status=success fixture_count={}",
        REQUIRED_MODEL_FIXTURE_IDS.len()
    );
}
