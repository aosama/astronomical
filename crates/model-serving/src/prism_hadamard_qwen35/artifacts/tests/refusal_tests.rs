//! Refusal contracts: every inventory violation is a typed refusal.

use super::super::{FoldedPackValidationError, FoldedPackValidator};
use super::pack_fixture::{remove_fixture_directory, write_fixture_pack};

fn validate_mutated(
    mutate: impl FnOnce(&mut super::pack_fixture::FixturePackMutation),
) -> (FoldedPackValidationError, std::path::PathBuf) {
    let fixture_directory = write_fixture_pack(mutate);
    let error = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect_err("every mutated fixture pack must be refused");
    (error, fixture_directory)
}

#[test]
fn should_refuse_a_missing_packed_tensor() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.drop_tensor("language_model.model.layers.0.mlp.down_proj.signs");
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::MissingPackedTensor { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_wrong_packed_dtype() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.retype_tensor("language_model.model.layers.0.mlp.down_proj.weight", "F32");
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::TensorDtypeMismatch { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_wrong_packed_shape() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.reshape_tensor(
            "language_model.model.layers.0.mlp.down_proj.weight",
            vec![128, 15],
        );
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::TensorShapeMismatch { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_wrong_auxiliary_shape() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.reshape_tensor("language_model.model.layers.0.linear_attn.A_log", vec![15]);
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::TensorShapeMismatch { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_wrong_auxiliary_dtype() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.retype_tensor(
            "language_model.model.layers.0.input_layernorm.weight",
            "F16",
        );
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::TensorDtypeMismatch { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_manifest_and_module_disagreement() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        // Remove one rotated tensor from the manifest so the module records
        // and the manifest no longer agree.
        let rotated = mutation.hadamard["prism.hadamard.weight_names"]
            .as_array_mut()
            .expect("the manifest should carry an array");
        rotated.remove(0);
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::ContractModuleDisagreement { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_an_unrecognized_language_tensor() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.tensor_declarations.push((
            "language_model.model.mystery.weight".to_owned(),
            "F32".to_owned(),
            vec![128],
        ));
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::UnrecognizedLanguageTensor { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_vision_tensors_without_a_declared_vision_component() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.tensor_declarations.push((
            "vision_tower.patch_conv.weight".to_owned(),
            "F32".to_owned(),
            vec![4, 4],
        ));
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::UnrecognizedLanguageTensor { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_module_block_disagreeing_with_the_manifest() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        // 1024 is itself a supported block size, so this isolates the
        // config-vs-manifest disagreement from the supported-size gate.
        mutation.config["modules"][2]["block"] = serde_json::json!(1024);
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::ModuleBlockMismatch { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_projection_width_not_divisible_by_the_hadamard_block() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.hadamard["prism.hadamard.block_size"] = serde_json::json!(1024);
        for packed_module in mutation.config["modules"]
            .as_array_mut()
            .expect("modules array")
        {
            packed_module["block"] = serde_json::json!(1024);
        }
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::ContractModuleDisagreement { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_module_path_outside_the_family_layout() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        mutation.config["modules"]
            .as_array_mut()
            .expect("modules array")
            .push(serde_json::json!({
                "path": "layers.0.mystery_proj",
                "block": 1024,
                "embedding": false,
                "dtype": "float16",
            }));
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::ContractModuleDisagreement { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_pack_missing_a_canonical_projection_module() {
    let (error, fixture_directory) = validate_mutated(|mutation| {
        let missing_module_path = "model.layers.0.mlp.gate_proj";
        let missing_module_tensor_stem = format!("language_model.{missing_module_path}");
        let missing_module_weight_name = format!("{missing_module_tensor_stem}.weight");
        mutation.config["modules"]
            .as_array_mut()
            .expect("modules array")
            .retain(|packed_module| packed_module["path"].as_str() != Some(missing_module_path));
        mutation.hadamard["prism.hadamard.weight_names"]
            .as_array_mut()
            .expect("rotated tensor names")
            .retain(|tensor_name| {
                tensor_name.as_str() != Some(missing_module_weight_name.as_str())
            });
        mutation
            .tensor_declarations
            .retain(|(tensor_name, _, _)| !tensor_name.starts_with(&missing_module_tensor_stem));
    });
    assert!(matches!(
        error,
        FoldedPackValidationError::ContractModuleDisagreement { .. }
    ));
    remove_fixture_directory(&fixture_directory);
}
