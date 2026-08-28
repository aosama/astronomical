use std::{collections::HashMap, time::Duration};

use astronomical_config::resolve_model_id;
use astronomical_experimental_aligned_expert_packs::{
    AlignedExpertPackTensorDescriptor, build_aligned_expert_pack_metal_io_descriptors,
    read_aligned_expert_pack_header, validate_aligned_expert_pack_header,
};
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, build_quantized_expert_layer_plan,
    build_quantized_expert_page_manifest_from_plan,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::{
    aligned_expert_pack_projection::{
        measure_bounded_reader_projections, measure_metal_io_projections, selected_expert_ids,
    },
    require_aligned_expert_pack_completion,
};

const ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN: usize = 8;
const PROMPT_PROCESSING_PREFILL_CHUNCK_TOKENS: usize = 2_048;
const DATA_PLANE_MLX_MEMORY_LIMIT_BYTES: usize = 8_000_000_000;
const MEASUREMENT_PAIR_COUNT: usize = 20;

#[derive(Clone, Copy)]
struct ExpertDataPlaneModel {
    configured_model_directory_name: &'static str,
    model_id: &'static str,
}

#[derive(Clone, Copy)]
struct ExpertDataPlaneScenario {
    scenario_name: &'static str,
    routed_token_count: usize,
    selected_expert_count: Option<usize>,
}

impl ExpertDataPlaneScenario {
    const fn generation() -> Self {
        Self {
            scenario_name: "generation",
            routed_token_count: 1,
            selected_expert_count: Some(ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN),
        }
    }

    const fn prompt_processing() -> Self {
        Self {
            scenario_name: "prompt_processing",
            routed_token_count: PROMPT_PROCESSING_PREFILL_CHUNCK_TOKENS,
            selected_expert_count: None,
        }
    }

    fn routed_page_slot_ids(self, selected_expert_count: usize) -> Vec<i32> {
        let routed_assignment_count = self
            .routed_token_count
            .checked_mul(ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN)
            .expect("routed assignment count should fit usize");
        (0..routed_assignment_count)
            .map(|routed_assignment_position| {
                i32::try_from(routed_assignment_position % selected_expert_count)
                    .expect("selected expert page slot should fit i32")
            })
            .collect()
    }
}

#[tokio::test]
#[ignore = "measures one Ornith generation layer through bounded and Metal gather-QMM paths"]
async fn should_measure_one_layer_generation_expert_data_plane() {
    require_aligned_expert_pack_completion(run_expert_data_plane_measurements(
        ExpertDataPlaneModel {
            configured_model_directory_name: super::ORNITH_OQ6_MODEL_ID,
            model_id: super::ORNITH_OQ6_PROVIDER_MODEL_ID,
        },
        &[ExpertDataPlaneScenario::generation()],
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures one Ornith prefill layer through bounded and Metal gather-QMM paths"]
async fn should_measure_one_layer_prompt_processing_expert_data_plane() {
    require_aligned_expert_pack_completion(run_expert_data_plane_measurements(
        ExpertDataPlaneModel {
            configured_model_directory_name: super::ORNITH_OQ6_MODEL_ID,
            model_id: super::ORNITH_OQ6_PROVIDER_MODEL_ID,
        },
        &[ExpertDataPlaneScenario::prompt_processing()],
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures exact oQ6e generation and prefill data planes in balanced order"]
async fn should_measure_oq6e_expert_data_plane_in_both_orders() {
    require_aligned_expert_pack_completion(run_expert_data_plane_measurements(
        ExpertDataPlaneModel {
            configured_model_directory_name: super::ORNITH_OQ6_MODEL_ID,
            model_id: super::ORNITH_OQ6_PROVIDER_MODEL_ID,
        },
        &[
            ExpertDataPlaneScenario::generation(),
            ExpertDataPlaneScenario::prompt_processing(),
        ],
    ))
    .await;
}

async fn run_expert_data_plane_measurements(
    expert_data_plane_model: ExpertDataPlaneModel,
    expert_data_plane_scenarios: &[ExpertDataPlaneScenario],
) {
    let _direct_mlx_guard = super::direct_mlx_test_guard().await;
    let model_directory = super::configured_model_directory_by_id(
        expert_data_plane_model.configured_model_directory_name,
    )
    .expect("the configured data-plane model should be installed");
    eprintln!(
        "[expert-storage-data-plane] status=progress phase=artifact_validation model_id={}",
        expert_data_plane_model.model_id
    );
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the configured data-plane artifact should validate");
    assert_eq!(
        resolve_model_id(
            expert_data_plane_model.model_id,
            &[validated_artifact.model_id()],
        ),
        validated_artifact.model_id()
    );
    let qwen3_5_config = validated_artifact.config().clone();
    let language_tensor_name_to_shard_file_name = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect::<HashMap<_, _>>();
    let layer_plan = build_quantized_expert_layer_plan(
        &model_directory,
        &language_tensor_name_to_shard_file_name,
        "language_model.model.layers.0.mlp",
        &qwen3_5_config,
    )
    .expect("the first decoder layer should have one validated affine expert plan");
    let aligned_expert_pack_path = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join(validated_artifact.revision())
        .join("layer-0.aligned-expert-pack");
    let aligned_expert_pack_header = read_aligned_expert_pack_header(&aligned_expert_pack_path)
        .expect("the prepared first-layer aligned pack should be readable");
    validate_aligned_expert_pack_header(
        &aligned_expert_pack_path,
        &aligned_expert_pack_header,
        &layer_plan,
        validated_artifact.model_id(),
        validated_artifact.revision(),
        0,
    )
    .expect("the prepared first-layer pack should match the validated source layer");
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DATA_PLANE_MLX_MEMORY_LIMIT_BYTES,
            DATA_PLANE_MLX_MEMORY_LIMIT_BYTES,
        )
        .expect("the data-plane MLX memory limits should be valid"),
    )
    .expect("the data-plane MLX runtime should initialize");

    for &expert_data_plane_scenario in expert_data_plane_scenarios {
        measure_expert_data_plane_scenario(
            &runtime,
            &aligned_expert_pack_path,
            &aligned_expert_pack_header.tensor_descriptors,
            &layer_plan,
            expert_data_plane_model.model_id,
            expert_data_plane_scenario,
        );
    }
}

fn measure_expert_data_plane_scenario(
    runtime: &MlxRuntime,
    aligned_expert_pack_path: &std::path::Path,
    tensor_descriptors: &[AlignedExpertPackTensorDescriptor],
    layer_plan: &astronomical_model_serving::QuantizedExpertLayerPlan,
    model_id: &str,
    expert_data_plane_scenario: ExpertDataPlaneScenario,
) {
    let selected_expert_count = expert_data_plane_scenario
        .selected_expert_count
        .unwrap_or(layer_plan.expert_capacity);
    let selected_expert_ids =
        selected_expert_ids(layer_plan.expert_capacity, selected_expert_count);
    let bounded_reader_page_manifest =
        build_quantized_expert_page_manifest_from_plan(layer_plan, &selected_expert_ids)
            .expect("selected experts should form one bounded reader page manifest");
    let selected_page_slot_ids =
        expert_data_plane_scenario.routed_page_slot_ids(selected_expert_count);
    let (metal_output_tensors, metal_load_ranges) = build_aligned_expert_pack_metal_io_descriptors(
        tensor_descriptors,
        layer_plan,
        &selected_expert_ids,
    )
    .expect("selected experts should produce Metal I/O descriptors");
    let tensor_name_to_metal_output_index = tensor_name_to_metal_output_indices(tensor_descriptors);

    eprintln!(
        "[expert-storage-data-plane] status=progress phase=warmup model_id={model_id} scenario={}",
        expert_data_plane_scenario.scenario_name
    );
    drop(measure_bounded_reader_projections(
        runtime,
        &bounded_reader_page_manifest,
        &selected_page_slot_ids,
        layer_plan,
    ));
    drop(measure_metal_io_projections(
        runtime,
        aligned_expert_pack_path,
        &metal_output_tensors,
        &metal_load_ranges,
        &tensor_name_to_metal_output_index,
        &selected_page_slot_ids,
        layer_plan,
    ));

    let mut bounded_reader_elapsed_measurements = Vec::with_capacity(MEASUREMENT_PAIR_COUNT);
    let mut metal_io_elapsed_measurements = Vec::with_capacity(MEASUREMENT_PAIR_COUNT);
    for measurement_pair_number in 1..=MEASUREMENT_PAIR_COUNT {
        let (
            bounded_reader_projection_outputs,
            bounded_reader_elapsed,
            metal_projection_outputs,
            metal_io_metrics,
            metal_io_elapsed,
        ) = if measurement_pair_number % 2 == 1 {
            let (bounded_reader_projection_outputs, bounded_reader_elapsed) =
                measure_bounded_reader_projections(
                    runtime,
                    &bounded_reader_page_manifest,
                    &selected_page_slot_ids,
                    layer_plan,
                );
            let (metal_projection_outputs, metal_io_metrics, metal_io_elapsed) =
                measure_metal_io_projections(
                    runtime,
                    aligned_expert_pack_path,
                    &metal_output_tensors,
                    &metal_load_ranges,
                    &tensor_name_to_metal_output_index,
                    &selected_page_slot_ids,
                    layer_plan,
                );
            (
                bounded_reader_projection_outputs,
                bounded_reader_elapsed,
                metal_projection_outputs,
                metal_io_metrics,
                metal_io_elapsed,
            )
        } else {
            let (metal_projection_outputs, metal_io_metrics, metal_io_elapsed) =
                measure_metal_io_projections(
                    runtime,
                    aligned_expert_pack_path,
                    &metal_output_tensors,
                    &metal_load_ranges,
                    &tensor_name_to_metal_output_index,
                    &selected_page_slot_ids,
                    layer_plan,
                );
            let (bounded_reader_projection_outputs, bounded_reader_elapsed) =
                measure_bounded_reader_projections(
                    runtime,
                    &bounded_reader_page_manifest,
                    &selected_page_slot_ids,
                    layer_plan,
                );
            (
                bounded_reader_projection_outputs,
                bounded_reader_elapsed,
                metal_projection_outputs,
                metal_io_metrics,
                metal_io_elapsed,
            )
        };
        assert_projection_parity(
            &bounded_reader_projection_outputs,
            &metal_projection_outputs,
        );
        bounded_reader_elapsed_measurements.push(bounded_reader_elapsed);
        metal_io_elapsed_measurements.push(metal_io_elapsed);
        eprintln!(
            "[expert-storage-data-plane] status=progress model_id={model_id} scenario={} pair={measurement_pair_number}/{MEASUREMENT_PAIR_COUNT} first_path={} bounded_reader_elapsed_ms={:.3} metal_io_elapsed_ms={:.3} requested_bytes={} command_count={}",
            expert_data_plane_scenario.scenario_name,
            if measurement_pair_number % 2 == 1 {
                "bounded_reader"
            } else {
                "metal_io"
            },
            bounded_reader_elapsed.as_secs_f64() * 1_000.0,
            metal_io_elapsed.as_secs_f64() * 1_000.0,
            metal_io_metrics.requested_byte_count,
            metal_io_metrics.command_count,
        );
    }

    let metal_io_faster_pair_count = bounded_reader_elapsed_measurements
        .iter()
        .zip(&metal_io_elapsed_measurements)
        .filter(|(bounded_reader_elapsed, metal_io_elapsed)| {
            metal_io_elapsed < bounded_reader_elapsed
        })
        .count();
    let bounded_reader_median = median_duration(&mut bounded_reader_elapsed_measurements);
    let metal_io_median = median_duration(&mut metal_io_elapsed_measurements);
    let metal_io_improvement_percent = (bounded_reader_median.as_secs_f64()
        - metal_io_median.as_secs_f64())
        / bounded_reader_median.as_secs_f64()
        * 100.0;
    eprintln!(
        "[expert-storage-data-plane] status=success model_id={model_id} scenario={} measurement_scope=isolated_expert_storage_data_plane isolates_expert_storage_data_plane=true file_page_state=warm pair_count={MEASUREMENT_PAIR_COUNT} metal_io_faster_pair_count={metal_io_faster_pair_count} bounded_reader_median_ms={:.3} metal_io_median_ms={:.3} metal_io_improvement_percent={metal_io_improvement_percent:.3}",
        expert_data_plane_scenario.scenario_name,
        bounded_reader_median.as_secs_f64() * 1_000.0,
        metal_io_median.as_secs_f64() * 1_000.0,
    );
}

fn assert_projection_parity(
    bounded_reader_projection_outputs: &HashMap<
        &'static str,
        astronomical_runtime_integration::MlxArray,
    >,
    metal_projection_outputs: &HashMap<&'static str, astronomical_runtime_integration::MlxArray>,
) {
    for projection_name in ["gate_proj", "up_proj", "down_proj"] {
        assert_eq!(
            bounded_reader_projection_outputs[projection_name]
                .to_vec_f32()
                .expect("the bounded reader projection should copy to the host for parity"),
            metal_projection_outputs[projection_name]
                .to_vec_f32()
                .expect("the Metal projection should copy to the host for parity"),
            "Metal I/O {projection_name} output must exactly match the bounded reader"
        );
    }
}

fn tensor_name_to_metal_output_indices(
    tensor_descriptors: &[AlignedExpertPackTensorDescriptor],
) -> HashMap<String, usize> {
    tensor_descriptors
        .iter()
        .enumerate()
        .map(|(output_tensor_index, tensor_descriptor)| {
            (
                format!(
                    "{}.{}",
                    tensor_descriptor.projection_name, tensor_descriptor.parameter_name
                ),
                output_tensor_index,
            )
        })
        .collect()
}

fn median_duration(elapsed_measurements: &mut [Duration]) -> Duration {
    elapsed_measurements.sort_unstable();
    let middle_measurement_index = elapsed_measurements.len() / 2;
    if elapsed_measurements.len().is_multiple_of(2) {
        (elapsed_measurements[middle_measurement_index - 1]
            + elapsed_measurements[middle_measurement_index])
            / 2
    } else {
        elapsed_measurements[middle_measurement_index]
    }
}
