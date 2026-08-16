use astronomical_model_serving::{
    LagunaArtifactValidator, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};

use super::artifact_support::SyntheticLagunaArtifact;

#[test]
fn should_treat_an_absent_router_correction_bias_as_the_canonical_zero_default() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::sparse_stacked();
    fixture.remove_tensor_completely("model.layers.0.mlp.e_score_correction_bias");
    fixture.write(model_directory.path());

    let artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("an absent optional router correction bias should validate");
    assert!(
        artifact
            .tensor_contract()
            .descriptor(&LagunaTensorId::Layer {
                layer_index: 0,
                role: LagunaLayerTensorRole::RouterCorrectionBias,
                component: LagunaTensorComponent::Weight,
            })
            .is_none()
    );
}
