use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

use astronomical_model_serving::{
    QuantizedExpertLayerPlan, QuantizedExpertPageManifest, load_quantized_expert_page,
};
use astronomical_runtime_integration::{
    MlxArray, MlxMetalExpertPackLoad, MlxMetalExpertPackLoadMetrics, MlxMetalExpertPackLoadRange,
    MlxMetalExpertPackOutputTensor, MlxRuntime,
};

pub(super) fn selected_expert_ids(
    expert_capacity: usize,
    selected_expert_count: usize,
) -> Vec<usize> {
    assert!(
        selected_expert_count > 0 && selected_expert_count <= expert_capacity,
        "selected expert count must fit the configured Ornith expert capacity"
    );
    if selected_expert_count == 1 {
        return vec![expert_capacity / 2];
    }
    (0..selected_expert_count)
        .map(|selected_expert_position| {
            selected_expert_position * (expert_capacity - 1) / (selected_expert_count - 1)
        })
        .collect()
}

pub(super) fn measure_bounded_reader_projections(
    runtime: &MlxRuntime,
    reference_page_manifest: &QuantizedExpertPageManifest,
    selected_page_slot_ids: &[i32],
    layer_plan: &QuantizedExpertLayerPlan,
) -> (HashMap<&'static str, MlxArray>, Duration) {
    let measurement_started_at = Instant::now();
    let mut bounded_reader_tensors =
        load_quantized_expert_page(runtime, reference_page_manifest, None)
            .expect("the bounded reader should load the selected expert reference tensors");
    let projection_outputs = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|projection_name| {
            (
                projection_name,
                run_projection(
                    runtime,
                    projection_name,
                    &mut bounded_reader_tensors,
                    selected_page_slot_ids,
                    layer_plan,
                ),
            )
        })
        .collect();
    (projection_outputs, measurement_started_at.elapsed())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measure_metal_io_projections(
    runtime: &MlxRuntime,
    aligned_expert_pack_path: &Path,
    metal_output_tensors: &[MlxMetalExpertPackOutputTensor],
    metal_load_ranges: &[MlxMetalExpertPackLoadRange],
    tensor_name_to_metal_output_index: &HashMap<String, usize>,
    selected_page_slot_ids: &[i32],
    layer_plan: &QuantizedExpertLayerPlan,
) -> (
    HashMap<&'static str, MlxArray>,
    MlxMetalExpertPackLoadMetrics,
    Duration,
) {
    let measurement_started_at = Instant::now();
    let metal_expert_pack_load = runtime
        .load_metal_expert_pack_ranges(
            aligned_expert_pack_path,
            metal_output_tensors,
            metal_load_ranges,
            None,
        )
        .expect("Metal I/O should submit the selected expert tensor ranges");
    let projection_outputs = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|projection_name| {
            (
                projection_name,
                run_metal_projection(
                    runtime,
                    projection_name,
                    &metal_expert_pack_load,
                    tensor_name_to_metal_output_index,
                    selected_page_slot_ids,
                    layer_plan,
                ),
            )
        })
        .collect();
    let completion_metrics = metal_expert_pack_load
        .wait_for_completion()
        .expect("the Metal I/O command buffer should complete successfully");
    (
        projection_outputs,
        completion_metrics,
        measurement_started_at.elapsed(),
    )
}

pub(super) fn run_projection(
    runtime: &MlxRuntime,
    projection_name: &str,
    bounded_reader_tensors: &mut HashMap<String, MlxArray>,
    selected_rhs_indices: &[i32],
    layer_plan: &QuantizedExpertLayerPlan,
) -> MlxArray {
    let packed_weight = bounded_reader_tensors
        .remove(&format!("{projection_name}.weight"))
        .expect("the bounded reader should return the packed weight");
    let quantization_scales = bounded_reader_tensors
        .remove(&format!("{projection_name}.scales"))
        .expect("the bounded reader should return quantization scales");
    let quantization_biases = bounded_reader_tensors
        .remove(&format!("{projection_name}.biases"))
        .expect("the bounded reader should return quantization biases");
    run_gather_quantized_matmul(
        runtime,
        projection_name,
        &packed_weight,
        &quantization_scales,
        &quantization_biases,
        selected_rhs_indices,
        layer_plan,
    )
}

pub(super) fn run_metal_projection(
    runtime: &MlxRuntime,
    projection_name: &str,
    metal_expert_pack_load: &MlxMetalExpertPackLoad,
    tensor_name_to_metal_output_index: &HashMap<String, usize>,
    selected_rhs_indices: &[i32],
    layer_plan: &QuantizedExpertLayerPlan,
) -> MlxArray {
    run_gather_quantized_matmul(
        runtime,
        projection_name,
        metal_expert_pack_load
            .output_array(tensor_name_to_metal_output_index[&format!("{projection_name}.weight")])
            .expect("the Metal I/O packed weight output should exist"),
        metal_expert_pack_load
            .output_array(tensor_name_to_metal_output_index[&format!("{projection_name}.scales")])
            .expect("the Metal I/O scales output should exist"),
        metal_expert_pack_load
            .output_array(tensor_name_to_metal_output_index[&format!("{projection_name}.biases")])
            .expect("the Metal I/O biases output should exist"),
        selected_rhs_indices,
        layer_plan,
    )
}

fn run_gather_quantized_matmul(
    runtime: &MlxRuntime,
    projection_name: &str,
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    selected_page_slot_ids: &[i32],
    layer_plan: &QuantizedExpertLayerPlan,
) -> MlxArray {
    assert_eq!(
        selected_page_slot_ids.len() % 8,
        0,
        "routed expert assignments should contain eight experts per token"
    );
    let routed_token_count = selected_page_slot_ids.len() / 8;
    let selected_page_slot_ids_array = runtime
        .array_from_i32(
            selected_page_slot_ids,
            &[
                1,
                i32::try_from(routed_token_count).expect("routed token count should fit i32"),
                8,
            ],
        )
        .expect("the compact selected expert IDs should form an MLX array");
    run_projection_arrays(
        runtime,
        projection_name,
        packed_weight,
        quantization_scales,
        quantization_biases,
        &selected_page_slot_ids_array,
        layer_plan,
    )
}

fn run_projection_arrays(
    runtime: &MlxRuntime,
    projection_name: &str,
    packed_weight: &MlxArray,
    quantization_scales: &MlxArray,
    quantization_biases: &MlxArray,
    selected_expert_ids_array: &MlxArray,
    layer_plan: &QuantizedExpertLayerPlan,
) -> MlxArray {
    let weight_tensor_source = layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| {
            tensor_source.projection_name == projection_name
                && tensor_source.parameter_name == "weight"
        })
        .expect("the validated layer plan should contain the projection weight");
    let activation_width = weight_tensor_source.full_shape[2]
        .checked_mul(32)
        .and_then(|packed_bit_count| {
            packed_bit_count.checked_div(weight_tensor_source.quantization_bits as usize)
        })
        .expect("the packed expert weight should imply a valid activation width");
    let routed_assignment_count = selected_expert_ids_array.element_count();
    assert_eq!(
        routed_assignment_count % 8,
        0,
        "selected expert indices should contain eight assignments per token"
    );
    let routed_token_count = routed_assignment_count / 8;
    let activation_element_count = activation_width
        .checked_mul(routed_token_count)
        .expect("projection activation element count should fit usize");
    let activations = runtime
        .array_from_f32(
            &vec![1.0_f32; activation_element_count],
            &[
                1,
                i32::try_from(routed_token_count).expect("routed token count should fit i32"),
                1,
                1,
                i32::try_from(activation_width).expect("activation width should fit i32"),
            ],
        )
        .expect("the deterministic projection activations should form an MLX array");
    let projection_output = runtime
        .gather_quantized_matmul_affine(
            &activations,
            packed_weight,
            quantization_scales,
            quantization_biases,
            None,
            Some(selected_expert_ids_array),
            true,
            weight_tensor_source.quantization_group_size,
            weight_tensor_source.quantization_bits,
            false,
        )
        .expect("the selected expert projection should build a gather_qmm graph");
    projection_output
        .evaluate()
        .expect("the selected expert projection should evaluate on the GPU");
    projection_output
}
