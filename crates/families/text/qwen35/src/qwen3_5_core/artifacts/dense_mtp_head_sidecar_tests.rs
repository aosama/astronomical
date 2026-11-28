use serde_json::{Value, json};

use super::{DenseMtpHeadSidecar, DenseMtpHeadSidecarError, dense_mtp_head_tensor_profiles};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_dense_qwen3_6_config;

fn head_config_document(model_config: &Qwen3_5Config) -> Value {
    super::super::dense_mtp_head_test_support::dense_mtp_head_config_document(model_config)
}

fn read(document: &Value) -> Result<Option<DenseMtpHeadSidecar>, DenseMtpHeadSidecarError> {
    DenseMtpHeadSidecar::from_config_bytes(
        document.to_string().as_bytes(),
        &frozen_dense_qwen3_6_config(),
    )
}

#[test]
fn should_read_a_head_that_fits_its_model() {
    let document = head_config_document(&frozen_dense_qwen3_6_config());
    let sidecar = read(&document)
        .expect("a fitting head should be read")
        .expect("a qwen3_5_mtp head is this kind");
    assert_eq!(sidecar.quantization_bits, 4);
    assert_eq!(sidecar.quantization_group_size, 64);
}

#[test]
fn should_pass_over_a_folder_holding_another_kind_of_head() {
    let mut document = head_config_document(&frozen_dense_qwen3_6_config());
    document
        .as_object_mut()
        .expect("an object")
        .remove("model_type");
    assert!(
        read(&document)
            .expect("another kind is not an error")
            .is_none()
    );
    document["model_type"] = json!("qwen4_exp_mtp");
    assert!(
        read(&document)
            .expect("another kind is not an error")
            .is_none()
    );
}

#[test]
fn should_refuse_a_head_sized_for_another_model() {
    let mut document = head_config_document(&frozen_dense_qwen3_6_config());
    document["text_config"]["hidden_size"] = json!(4096 + 64);
    assert!(matches!(
        read(&document),
        Err(DenseMtpHeadSidecarError::ModelMismatch {
            field: "hidden_size",
            ..
        })
    ));
}

#[test]
fn should_refuse_quantization_mlx_cannot_serve() {
    let mut document = head_config_document(&frozen_dense_qwen3_6_config());
    document["quantization"] = json!({"group_size": 64, "bits": 7});
    assert!(matches!(
        read(&document),
        Err(DenseMtpHeadSidecarError::UnsupportedQuantization { bits: 7, .. })
    ));
    document
        .as_object_mut()
        .expect("an object")
        .remove("quantization");
    assert!(matches!(
        read(&document),
        Err(DenseMtpHeadSidecarError::QuantizationMissing)
    ));
}

#[test]
fn should_refuse_a_head_with_embeddings_of_its_own() {
    let mut document = head_config_document(&frozen_dense_qwen3_6_config());
    document["text_config"]["mtp_use_dedicated_embeddings"] = json!(true);
    assert!(matches!(
        read(&document),
        Err(DenseMtpHeadSidecarError::DedicatedEmbeddings)
    ));
}

#[test]
fn should_refuse_a_config_that_is_not_json() {
    let refusal =
        DenseMtpHeadSidecar::from_config_bytes(b"{not json", &frozen_dense_qwen3_6_config());
    assert!(matches!(
        refusal,
        Err(DenseMtpHeadSidecarError::Deserialize(_))
    ));
}

/// Ternary Bonsai 2 9B's head ships exactly 31 tensors: the fusion
/// projection (3), one full-attention layer with its MLP (21), and five norms.
#[test]
fn should_list_every_head_tensor_with_the_fusion_over_twice_the_hidden_width() {
    let model_config = frozen_dense_qwen3_6_config();
    let head_config = DenseMtpHeadSidecar {
        quantization_bits: 4,
        quantization_group_size: 64,
    }
    .head_config(&model_config);
    let profiles =
        dense_mtp_head_tensor_profiles(&head_config).expect("the head's tensors should list");
    assert_eq!(profiles.len(), 31);
    let fusion_weight = profiles
        .iter()
        .find(|profile| profile.name == "fc.weight")
        .expect("the fusion projection is listed");
    let packed_input_words = model_config.hidden_size() as usize * 2 * 4 / 32;
    assert_eq!(
        fusion_weight.shape,
        vec![model_config.hidden_size() as usize, packed_input_words]
    );
}

#[test]
fn should_quantize_every_head_module_with_the_heads_own_profile() {
    let model_config = frozen_dense_qwen3_6_config();
    let head_config = DenseMtpHeadSidecar {
        quantization_bits: 4,
        quantization_group_size: 64,
    }
    .head_config(&model_config);
    for module_name in ["fc", "layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"] {
        let profile = head_config.quantization_profile_for_module(module_name);
        assert_eq!((profile.bits, profile.group_size), (4, 64), "{module_name}");
    }
}

#[test]
fn should_refuse_a_head_whose_dimensions_cannot_be_checked() {
    let mut document = head_config_document(&frozen_dense_qwen3_6_config());
    document
        .as_object_mut()
        .expect("an object")
        .remove("text_config");
    assert!(matches!(
        read(&document),
        Err(DenseMtpHeadSidecarError::DimensionsMissing)
    ));
}
