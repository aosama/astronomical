//! Unit tests for reading a folder's add-ons and checking them against the model.

use decision_contract::decision_adapters_file::{LinearAdapter, decision_adapters_bytes};
use serial_test::file_serial;

use super::*;
use model_primitives::synthetic_weights_test_support::{
    SYNTHETIC_QUANTIZATION_GROUP_SIZE, SyntheticWeights, initialize_test_runtime,
    load_synthetic_weights,
};

fn output_projection_adapter(input_width: usize, output_width: usize) -> DecisionAdapters {
    DecisionAdapters {
        adapters: vec![LinearAdapter {
            target: "layers.0.attn.Wo".to_owned(),
            rank: 1,
            input_width,
            output_width,
            down_values: vec![0.5; input_width],
            up_values: vec![0.0; output_width],
        }],
    }
}

fn output_projection_target() -> AdapterTarget {
    AdapterTarget {
        name: "layers.0.attn.Wo".to_owned(),
        input_width: 64,
        output_width: 32,
    }
}

#[test]
fn should_read_no_add_ons_from_a_folder_without_the_file() {
    let model_directory = tempfile::tempdir().unwrap();

    assert!(
        read_decision_adapters(model_directory.path())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn should_read_the_add_ons_a_folder_carries() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_ADAPTERS_FILE_NAME),
        decision_adapters_bytes(&output_projection_adapter(64, 32)).unwrap(),
    )
    .unwrap();

    assert_eq!(
        read_decision_adapters(model_directory.path()).unwrap(),
        output_projection_adapter(64, 32)
    );
}

#[test]
fn should_refuse_to_load_with_an_add_on_file_it_cannot_use() {
    let model_directory = tempfile::tempdir().unwrap();
    std::fs::write(
        model_directory.path().join(DECISION_ADAPTERS_FILE_NAME),
        b"Wherefore art thou Romeo",
    )
    .unwrap();

    assert!(read_decision_adapters(model_directory.path()).is_err());
}

#[test]
fn should_accept_add_ons_that_match_their_layers() {
    assert!(
        check_adapter_targets(
            &output_projection_adapter(64, 32),
            &[output_projection_target()]
        )
        .is_ok()
    );
}

#[test]
fn should_refuse_an_add_on_for_a_layer_the_model_does_not_have() {
    let refusal = check_adapter_targets(&output_projection_adapter(64, 32), &[]).unwrap_err();

    assert!(
        refusal.to_string().contains("layers.0.attn.Wo"),
        "{refusal:?}"
    );
}

#[test]
fn should_refuse_an_add_on_whose_widths_differ_from_its_layer() {
    let refusal = check_adapter_targets(
        &output_projection_adapter(32, 64),
        &[output_projection_target()],
    )
    .unwrap_err();

    assert!(
        refusal.to_string().contains("maps 32 inputs"),
        "{refusal:?}"
    );
}

#[test]
#[file_serial]
fn should_read_a_quantized_layers_widths_from_its_stored_tensors() {
    let runtime = initialize_test_runtime();
    let mut synthetic_weights = SyntheticWeights::new();
    synthetic_weights.push_quantized(&runtime, "layers.0.attn.Wo", [32, 64]);
    let tensors = load_synthetic_weights(&runtime, &synthetic_weights.named_tensors);

    let target = AdapterTarget::of_quantized_linear(
        &tensors,
        "layers.0.attn.Wo".to_owned(),
        SYNTHETIC_QUANTIZATION_GROUP_SIZE as usize,
    )
    .unwrap();

    assert_eq!(target, output_projection_target());
    assert!(
        AdapterTarget::of_quantized_linear(
            &tensors,
            "layers.1.attn.Wo".to_owned(),
            SYNTHETIC_QUANTIZATION_GROUP_SIZE as usize
        )
        .is_err()
    );
}
