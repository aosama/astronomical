use std::path::Path;

use serde_json::{Value, json};

use super::{DenseMtpHeadLoadError, locate_dense_mtp_head};
use crate::qwen3_5_core::artifacts::dense_mtp_head_sidecar::DenseMtpHeadSidecarError;
use crate::qwen3_5_core::artifacts::dense_mtp_head_test_support::dense_mtp_head_config_document;
use crate::qwen3_5_core::frozen_config_test_support::{
    frozen_dense_qwen3_6_config, frozen_ornith_1_0_config,
};

fn fitting_head_config() -> Value {
    dense_mtp_head_config_document(&frozen_dense_qwen3_6_config())
}

/// A weights file whose header declares `tensor_names`, each a one-element
/// float16 tensor: locating a head reads only the header.
fn header_only_safetensors(tensor_names: &[&str]) -> Vec<u8> {
    let mut header_entries = serde_json::Map::new();
    for (tensor_index, tensor_name) in tensor_names.iter().enumerate() {
        header_entries.insert(
            (*tensor_name).to_owned(),
            json!({"dtype": "F16", "shape": [1], "data_offsets": [tensor_index * 2, tensor_index * 2 + 2]}),
        );
    }
    let header_bytes = Value::Object(header_entries).to_string().into_bytes();
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend(header_bytes);
    file_bytes.extend(std::iter::repeat_n(0_u8, tensor_names.len() * 2));
    file_bytes
}

fn write_head(model_directory: &Path, head_config: &[u8], tensor_names: &[&str]) {
    let head_directory = model_directory.join("mtp");
    std::fs::create_dir_all(&head_directory).expect("the head folder should be created");
    std::fs::write(head_directory.join("config.json"), head_config)
        .expect("the head config should be written");
    std::fs::write(
        head_directory.join("model.safetensors"),
        header_only_safetensors(tensor_names),
    )
    .expect("the head weights should be written");
}

#[test]
fn should_find_no_head_where_the_pack_ships_none() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    let located = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect("a pack without a head is not an error");
    assert!(located.is_none());
}

#[test]
fn should_pass_over_a_head_folder_of_another_kind() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    let other_kind = json!({"source_file": "Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf"});
    write_head(
        model_directory.path(),
        other_kind.to_string().as_bytes(),
        &["norm.weight"],
    );
    let located = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect("another kind of head is not an error");
    assert!(located.is_none());
}

#[test]
fn should_locate_a_fitting_head_and_read_its_tensor_names() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    write_head(
        model_directory.path(),
        fitting_head_config().to_string().as_bytes(),
        &["fc.weight", "norm.weight"],
    );
    let located = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect("a fitting head should be located")
        .expect("the folder holds a dense head");
    assert_eq!(located.sidecar.quantization_bits, 4);
    assert!(located.shipped_tensor_names.contains("fc.weight"));
    assert_eq!(located.shipped_tensor_names.len(), 2);
}

#[test]
fn should_refuse_a_head_sized_for_another_model() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    let mut head_config = fitting_head_config();
    head_config["text_config"]["vocab_size"] = json!(1024);
    write_head(
        model_directory.path(),
        head_config.to_string().as_bytes(),
        &["fc.weight"],
    );
    let refusal = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect_err("a head for another model must be refused");
    assert!(matches!(
        refusal,
        DenseMtpHeadLoadError::Configuration(DenseMtpHeadSidecarError::ModelMismatch {
            field: "vocab_size",
            ..
        })
    ));
}

#[test]
fn should_refuse_a_configuration_too_large_to_be_a_head() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    let oversized_config = vec![b' '; 1024 * 1024 + 1];
    write_head(model_directory.path(), &oversized_config, &["fc.weight"]);
    let refusal = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect_err("an oversized configuration must be refused");
    assert!(matches!(
        refusal,
        DenseMtpHeadLoadError::ConfigurationTooLarge
    ));
}

#[test]
fn should_refuse_a_head_whose_weights_file_is_missing() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    write_head(
        model_directory.path(),
        fitting_head_config().to_string().as_bytes(),
        &["fc.weight"],
    );
    std::fs::remove_file(model_directory.path().join("mtp/model.safetensors"))
        .expect("the fixture weights should be removable");
    let refusal = locate_dense_mtp_head(&frozen_dense_qwen3_6_config(), model_directory.path())
        .expect_err("a head without weights must be refused");
    assert!(matches!(refusal, DenseMtpHeadLoadError::Header { .. }));
}

/// A mixture-of-experts model's state is not what a dense head joins, so a
/// head folder beside one is not this kind's to load.
#[test]
fn should_pass_over_a_head_beside_a_model_that_is_not_dense() {
    let model_directory = tempfile::tempdir().expect("a scratch folder");
    let routed_model_config = frozen_ornith_1_0_config();
    write_head(
        model_directory.path(),
        dense_mtp_head_config_document(&routed_model_config)
            .to_string()
            .as_bytes(),
        &["fc.weight"],
    );
    let located = locate_dense_mtp_head(&routed_model_config, model_directory.path())
        .expect("a routed model is not an error");
    assert!(located.is_none());
}
