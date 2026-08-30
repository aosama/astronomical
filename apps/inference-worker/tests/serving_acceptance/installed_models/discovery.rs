//! Preflights the e2e roles remaining acceptance journeys require.
//!
//! Expensive journeys should fail before loading MLX when a required fixture was
//! removed or renamed. FLUX absence is not a preflight failure; FLUX tests fail
//! closed at their own discovery.

#[test]
#[ignore = "reads Development model_directories to preflight required e2e roles"]
fn should_discover_required_installed_models() {
    let discovered_models = crate::support::configured_discovered_models();
    let required_model_ids = crate::support::required_e2e_test_model_ids();
    let mut missing_model_ids = Vec::new();

    for required_model_id in required_model_ids {
        eprintln!("[model-fixture-discovery] status=progress model={required_model_id}");
        if !discovered_models
            .iter()
            .any(|discovered_model| discovered_model.model_id == required_model_id)
        {
            missing_model_ids.push(required_model_id);
        }
    }

    assert!(
        missing_model_ids.is_empty(),
        "Development model_directories must discover every required e2e role; missing={missing_model_ids:?}"
    );

    let flux_model_id = crate::support::flux2_klein_model_id();
    let flux_is_discovered = discovered_models
        .iter()
        .any(|discovered_model| discovered_model.model_id == flux_model_id);
    if flux_is_discovered {
        eprintln!(
            "[model-fixture-discovery] status=progress model={flux_model_id} optional=present"
        );
    } else {
        eprintln!(
            "[model-fixture-discovery] status=progress model={flux_model_id} optional=absent"
        );
    }

    eprintln!(
        "[model-fixture-discovery] status=success required_role_count={}",
        required_model_ids.len()
    );
}
