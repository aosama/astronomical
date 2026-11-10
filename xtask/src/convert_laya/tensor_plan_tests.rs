//! Unit contracts for what becomes of each checkpoint tensor: its MLX name,
//! whether it is quantized, and the shapes written for it.

use super::super::safetensors_file::{TensorElementType, TensorRecord};
use super::{TensorTreatment, planned_output_tensors, tensor_treatment};

fn kept_as(output_name: &str) -> TensorTreatment {
    TensorTreatment::KeptAsHalfPrecision {
        output_name: output_name.to_owned(),
    }
}

fn quantized_as(module_name: &str) -> TensorTreatment {
    TensorTreatment::Quantized {
        module_name: module_name.to_owned(),
    }
}

fn source_record(tensor_name: &str, shape: &[u64]) -> TensorRecord {
    TensorRecord {
        tensor_name: tensor_name.to_owned(),
        element_type: TensorElementType::Float16,
        shape: shape.to_vec(),
        data_start: 0,
        byte_count: shape.iter().product::<u64>() * 2,
    }
}

#[test]
fn should_quantize_the_encoder_and_head_projections() {
    for module_name in [
        "encoder.layers.0.attn.Wo",
        "encoder.layers.27.attn.Wqkv",
        "encoder.layers.3.mlp.Wi",
        "encoder.layers.3.mlp.Wo",
        "head.layers.0.linear1",
        "head.layers.1.linear2",
        "head.layers.1.self_attn.out_proj",
    ] {
        assert_eq!(
            tensor_treatment(&format!("{module_name}.weight"), &[64, 64]),
            quantized_as(module_name)
        );
    }
}

#[test]
fn should_rename_the_packed_attention_projection_and_quantize_its_weight() {
    assert_eq!(
        tensor_treatment("head.layers.0.self_attn.in_proj_weight", &[96, 32]),
        quantized_as("head.layers.0.self_attn.in_proj")
    );
    assert_eq!(
        tensor_treatment("head.layers.0.self_attn.in_proj_bias", &[96]),
        kept_as("head.layers.0.self_attn.in_proj.bias")
    );
}

#[test]
fn should_keep_embeddings_norms_and_the_scorer_in_half_precision() {
    assert_eq!(
        tensor_treatment("encoder.embeddings.tok_embeddings.weight", &[128, 64]),
        kept_as("encoder.embeddings.tok_embeddings.weight")
    );
    assert_eq!(
        tensor_treatment("encoder.layers.0.mlp_norm.weight", &[64]),
        kept_as("encoder.layers.0.mlp_norm.weight")
    );
    assert_eq!(
        tensor_treatment("type_emb.weight", &[3, 64]),
        kept_as("type_emb.weight")
    );
    assert_eq!(
        tensor_treatment("head.layers.0.linear1.bias", &[64]),
        kept_as("head.layers.0.linear1.bias")
    );
}

#[test]
fn should_move_sequential_children_under_layers_without_quantizing_them() {
    assert_eq!(
        tensor_treatment("scorer.1.weight", &[64, 64]),
        kept_as("scorer.layers.1.weight")
    );
    assert_eq!(
        tensor_treatment("scorer.3.bias", &[1]),
        kept_as("scorer.layers.3.bias")
    );
    assert_eq!(
        tensor_treatment("act_head.0.weight", &[32, 68]),
        kept_as("act_head.layers.0.weight")
    );
}

#[test]
fn should_drop_the_temperature_tensor() {
    assert_eq!(
        tensor_treatment("temperature", &[3]),
        TensorTreatment::Dropped
    );
    assert!(
        planned_output_tensors(&source_record("temperature", &[3]))
            .expect("a dropped tensor plans nothing")
            .is_empty()
    );
}

#[test]
fn should_not_quantize_a_one_dimensional_tensor_named_like_a_projection() {
    assert_eq!(
        tensor_treatment("encoder.layers.0.attn.Wo.weight", &[64]),
        kept_as("encoder.layers.0.attn.Wo.weight")
    );
}

#[test]
fn should_plan_a_packed_weight_then_scales_then_biases_for_a_quantized_projection() {
    let planned_tensors = planned_output_tensors(&source_record(
        "encoder.layers.0.mlp.Wo.weight",
        &[1024, 2624],
    ))
    .expect("a projection with whole groups should plan");

    let planned_shapes: Vec<(&str, TensorElementType, &[u64])> = planned_tensors
        .iter()
        .map(|planned_tensor| {
            (
                planned_tensor.tensor_name.as_str(),
                planned_tensor.element_type,
                planned_tensor.shape.as_slice(),
            )
        })
        .collect();
    assert_eq!(
        planned_shapes,
        [
            (
                "encoder.layers.0.mlp.Wo.weight",
                TensorElementType::Unsigned32,
                [1024, 656].as_slice()
            ),
            (
                "encoder.layers.0.mlp.Wo.scales",
                TensorElementType::Float16,
                [1024, 82].as_slice()
            ),
            (
                "encoder.layers.0.mlp.Wo.biases",
                TensorElementType::Float16,
                [1024, 82].as_slice()
            ),
        ]
    );
}

#[test]
fn should_refuse_a_projection_whose_inputs_do_not_fill_whole_groups() {
    let plan_outcome =
        planned_output_tensors(&source_record("encoder.layers.0.attn.Wo.weight", &[8, 48]));

    assert!(matches!(plan_outcome, Err(reason) if reason.contains("48 inputs")));
}
