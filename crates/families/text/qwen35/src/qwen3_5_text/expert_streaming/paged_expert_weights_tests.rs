//! Beside-source coverage for the plan checks that guard building a streamed
//! expert layer: rotation consistency, tensor-source lookup and the codebook
//! bit width. Taking the arrays themselves needs the graphics runtime and is
//! covered by the expert paging lane.

use std::path::PathBuf;

use engine_core::expert_paging::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
};
use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;

use super::{
    ExpertPagingError, codebook_bit_width_of, reject_declared_jangtq2_rotation,
    require_projection_tensor_source,
};

fn tensor_source(
    projection_name: &str,
    parameter_name: &str,
    quantization_bits: i32,
) -> QuantizedTensorSource {
    QuantizedTensorSource {
        tensor_name: format!("{projection_name}.{parameter_name}"),
        projection_name: projection_name.to_owned(),
        parameter_name: parameter_name.to_owned(),
        quantization_bits,
        quantization_group_size: 32,
        source_file: PathBuf::from("projection-plan-fixture.safetensors"),
        source_file_size_bytes: 2_048,
        dtype: SafetensorsDtype::Uint32,
        full_shape: vec![6, 3, 2],
        tensor_payload_offset: 128,
        bytes_per_expert: 24,
        expert_stride_bytes: 24,
        expert_capacity: 6,
    }
}

fn layer_plan_with(
    tensor_sources: Vec<QuantizedTensorSource>,
    jangtq2_projection_rotations: Vec<(String, Jangtq2CodebookRotation)>,
) -> QuantizedExpertLayerPlan {
    QuantizedExpertLayerPlan {
        layer_prefix: "projection-plan.layers.2".to_owned(),
        tensor_sources,
        jangtq2_projection_rotations,
        expert_capacity: 6,
        quantization_bits: 2,
        quantization_group_size: 32,
        quantization_mode: QuantizationMode::Affine,
    }
}

fn error_description(paging_error: ExpertPagingError) -> String {
    match paging_error {
        ExpertPagingError::InvalidPagingPlan { description }
        | ExpertPagingError::Runtime { description } => description,
        other_error => panic!("unexpected paging error: {other_error}"),
    }
}

#[test]
fn should_accept_a_projection_that_declares_no_rotation() {
    let layer_plan = layer_plan_with(Vec::new(), Vec::new());

    assert!(reject_declared_jangtq2_rotation(&layer_plan, "gate_proj", "native").is_ok());
}

#[test]
fn should_ignore_a_rotation_declared_for_another_projection() {
    let layer_plan = layer_plan_with(
        Vec::new(),
        vec![("down_proj".to_owned(), Jangtq2CodebookRotation::Hadamard32)],
    );

    assert!(reject_declared_jangtq2_rotation(&layer_plan, "gate_proj", "affine").is_ok());
}

#[test]
fn should_reject_a_declared_rotation_naming_the_storage_kind() {
    let layer_plan = layer_plan_with(
        Vec::new(),
        vec![("gate_proj".to_owned(), Jangtq2CodebookRotation::NoRotation)],
    );

    let rejection = reject_declared_jangtq2_rotation(&layer_plan, "gate_proj", "affine")
        .expect_err("a declared rotation must be refused");

    assert_eq!(
        error_description(rejection),
        "affine gate_proj projection unexpectedly declares a jangtq2 rotation"
    );
}

#[test]
fn should_find_the_source_of_the_named_projection_and_parameter() {
    let layer_plan = layer_plan_with(
        vec![
            tensor_source("gate_proj", "weight", 4),
            tensor_source("up_proj", "weight", 8),
            tensor_source("up_proj", "scales", 16),
        ],
        Vec::new(),
    );

    let source = require_projection_tensor_source(&layer_plan, "up_proj", "weight", "quantization")
        .expect("the up projection's weight source exists");

    assert_eq!(source.quantization_bits, 8);
}

#[test]
fn should_name_the_missing_metadata_when_no_source_matches() {
    let layer_plan = layer_plan_with(vec![tensor_source("gate_proj", "weight", 4)], Vec::new());

    let missing =
        require_projection_tensor_source(&layer_plan, "gate_proj", "tq2_packed", "codebook")
            .expect_err("no packed codebook tensor is planned");

    assert_eq!(
        error_description(missing),
        "layer plan is missing gate_proj codebook metadata"
    );
}

#[test]
fn should_not_match_a_source_of_the_right_parameter_on_another_projection() {
    let layer_plan = layer_plan_with(vec![tensor_source("down_proj", "weight", 4)], Vec::new());

    assert!(
        require_projection_tensor_source(&layer_plan, "gate_proj", "weight", "quantization")
            .is_err()
    );
}

#[test]
fn should_read_the_codebook_bit_width_from_the_packed_source() {
    let packed_source = tensor_source("gate_proj", "tq2_packed", 2);

    assert_eq!(
        codebook_bit_width_of(&packed_source, "gate_proj").ok(),
        Some(2)
    );
}

#[test]
fn should_accept_a_zero_codebook_bit_width() {
    let packed_source = tensor_source("gate_proj", "tq2_packed", 0);

    assert_eq!(
        codebook_bit_width_of(&packed_source, "gate_proj").ok(),
        Some(0)
    );
}

#[test]
fn should_reject_a_negative_codebook_bit_width() {
    let packed_source = tensor_source("up_proj", "tq2_packed", -1);

    let invalid_bits = codebook_bit_width_of(&packed_source, "up_proj")
        .expect_err("a negative width does not fit an unsigned bit width");

    assert_eq!(
        error_description(invalid_bits),
        "layer plan has invalid up_proj codebook bits"
    );
}
