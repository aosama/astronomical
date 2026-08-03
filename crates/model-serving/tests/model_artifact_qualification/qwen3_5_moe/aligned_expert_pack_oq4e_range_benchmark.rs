//! Model-resident A/B diagnostic for aligned-pack Metal I/O range coalescing.
//!
//! This isolates one complete affine expert layer so both variants transfer identical bytes and
//! execute identical gather-QMM work. It explains command-encoding behavior; the separate
//! 1,024-input and 512-output probe remains the representative production performance boundary.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use astronomical_ipc_protocol::ExpertStorageFormat;
use astronomical_model_serving::{
    QuantizationMode, Qwen3_5ArtifactValidator, Qwen3_5Model, RequestDecoderStateStack,
    build_aligned_expert_pack_metal_io_descriptors, build_quantized_expert_layer_plan,
    read_aligned_expert_pack_header, validate_aligned_expert_pack_header,
};
use astronomical_runtime_integration::{
    MlxArray, MlxMemoryLimits, MlxMetalExpertPackLoadMetrics, MlxMetalExpertPackLoadRange,
    MlxMetalExpertPackOutputTensor, MlxRuntime,
};
use tokio::time::timeout;

use super::aligned_expert_pack_projection::measure_metal_io_projections;

const OQ4E_MTP_MODEL_DIRECTORY_NAME: &str = "Qwen3.6-35B-A3B-oQ4e-mtp";
const CONFIGURED_MLX_MEMORY_LIMIT_BYTES: usize = 10_000_000_000;
const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(115);
const MEASUREMENT_PAIR_COUNT: usize = 5;

#[tokio::test]
#[ignore = "loads oQ4e with a configured 10 GB MLX limit and compares Metal I/O range plans"]
async fn should_measure_oq4e_coalesced_metal_io_ranges_with_configured_ten_gb_limit() {
    timeout(BENCHMARK_TIMEOUT, run_oq4e_range_coalescing_benchmark())
        .await
        .expect("the configured-10-GB oQ4e range benchmark should finish within 115 seconds");
}

async fn run_oq4e_range_coalescing_benchmark() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let benchmark_started_at = Instant::now();
    let model_directory = crate::common::configured_model_directory_by_id(
        OQ4E_MTP_MODEL_DIRECTORY_NAME,
    )
    .expect("the local Qwen3.6 oQ4e MTP model should be discoverable from configured roots");

    eprintln!("[oq4e-10gb-ranges] status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the local oQ4e MTP artifact should validate");
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
        QuantizationMode::Affine,
    )
    .expect("the first oQ4e layer should have one validated affine expert plan");
    let aligned_expert_pack_path = model_directory
        .join(".astronomical-aligned-expert-packs")
        .join(validated_artifact.revision())
        .join("layer-0.aligned-expert-pack");
    let aligned_expert_pack_header = read_aligned_expert_pack_header(&aligned_expert_pack_path)
        .expect("the prepared oQ4e layer-zero aligned pack should be readable");
    validate_aligned_expert_pack_header(
        &aligned_expert_pack_path,
        &aligned_expert_pack_header,
        &layer_plan,
        validated_artifact.model_id(),
        validated_artifact.revision(),
        0,
    )
    .expect("the prepared oQ4e layer-zero aligned pack should match the validated artifact");

    let constrained_mlx_memory_limits = MlxMemoryLimits::new(
        CONFIGURED_MLX_MEMORY_LIMIT_BYTES,
        CONFIGURED_MLX_MEMORY_LIMIT_BYTES,
    )
    .expect("the 10 GB MLX limits should be valid");
    eprintln!(
        "[oq4e-10gb-ranges] status=progress phase=model_load configured_memory_limit_bytes={CONFIGURED_MLX_MEMORY_LIMIT_BYTES} residency=automatic"
    );
    let runtime = MlxRuntime::initialize(constrained_mlx_memory_limits)
        .expect("the constrained MLX runtime should initialize");
    let qwen3_5_model = Qwen3_5Model::load(runtime, validated_artifact, &model_directory, false)
        .expect(
            "the oQ4e model should load with automatic residency and a configured 10 GB MLX limit",
        );
    assert_eq!(
        qwen3_5_model.runtime().memory_limits(),
        constrained_mlx_memory_limits
    );
    assert_eq!(
        qwen3_5_model.expert_storage_format(),
        ExpertStorageFormat::AstronomicalAligned,
        "the A/B benchmark must exercise the prepared aligned pack used by production paging"
    );
    let model_load_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the model-resident runtime should report its MLX memory snapshot");
    eprintln!("[oq4e-10gb-ranges] status=progress phase=model_activation prefill_chunck_tokens=1");
    let mut request_decoder_state = RequestDecoderStateStack::empty_from_config(&qwen3_5_config);
    let model_activation_started_at = Instant::now();
    qwen3_5_model
        .prefill_chunck(&[198], 0, &mut request_decoder_state)
        .expect("the oQ4e model should execute one paged prefill with the configured MLX limit");
    let model_activation_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the activated model should report its MLX memory snapshot");
    eprintln!(
        "[oq4e-10gb-ranges] status=progress phase=model_activated elapsed_seconds={:.3} active_memory_bytes={} peak_memory_bytes={}",
        model_activation_started_at.elapsed().as_secs_f64(),
        model_activation_memory_snapshot.active_memory_bytes(),
        model_activation_memory_snapshot.peak_memory_bytes(),
    );

    let selected_expert_ids = (0..layer_plan.expert_capacity).collect::<Vec<_>>();
    let selected_page_slot_ids = (0..selected_expert_ids.len())
        .map(|selected_page_slot| {
            i32::try_from(selected_page_slot)
                .expect("the complete selected-expert page should fit i32 indices")
        })
        .collect::<Vec<_>>();
    let (metal_output_tensors, coalesced_metal_load_ranges) =
        build_aligned_expert_pack_metal_io_descriptors(
            &aligned_expert_pack_header.tensor_descriptors,
            &layer_plan,
            &selected_expert_ids,
        )
        .expect("the complete oQ4e expert layer should produce coalesced Metal I/O descriptors");
    let legacy_metal_load_ranges = split_coalesced_ranges_into_legacy_ranges(
        &coalesced_metal_load_ranges,
        &aligned_expert_pack_header.tensor_descriptors,
    );
    let tensor_name_to_metal_output_index = aligned_expert_pack_header
        .tensor_descriptors
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
        .collect::<HashMap<_, _>>();
    assert_eq!(coalesced_metal_load_ranges.len(), 9);
    assert_eq!(
        legacy_metal_load_ranges.len(),
        aligned_expert_pack_header.tensor_descriptors.len() * selected_expert_ids.len()
    );

    eprintln!(
        "[oq4e-10gb-ranges] status=progress phase=warmup selected_expert_count={} legacy_range_count={} coalesced_range_count={}",
        selected_expert_ids.len(),
        legacy_metal_load_ranges.len(),
        coalesced_metal_load_ranges.len(),
    );
    let _legacy_warmup = measure_metal_io_projections(
        qwen3_5_model.runtime(),
        &aligned_expert_pack_path,
        &metal_output_tensors,
        &legacy_metal_load_ranges,
        &tensor_name_to_metal_output_index,
        &selected_page_slot_ids,
        &layer_plan,
    );
    let _coalesced_warmup = measure_metal_io_projections(
        qwen3_5_model.runtime(),
        &aligned_expert_pack_path,
        &metal_output_tensors,
        &coalesced_metal_load_ranges,
        &tensor_name_to_metal_output_index,
        &selected_page_slot_ids,
        &layer_plan,
    );

    let mut legacy_measurements = Vec::with_capacity(MEASUREMENT_PAIR_COUNT);
    let mut coalesced_measurements = Vec::with_capacity(MEASUREMENT_PAIR_COUNT);
    for measurement_pair_index in 0..MEASUREMENT_PAIR_COUNT {
        let coalesced_runs_first = measurement_pair_index % 2 == 1;
        eprintln!(
            "[oq4e-10gb-ranges] status=progress phase=measurement pair={}/{} first_variant={}",
            measurement_pair_index + 1,
            MEASUREMENT_PAIR_COUNT,
            if coalesced_runs_first {
                "coalesced"
            } else {
                "legacy"
            },
        );
        let first_measurement = measure_variant(
            qwen3_5_model.runtime(),
            &aligned_expert_pack_path,
            &metal_output_tensors,
            if coalesced_runs_first {
                &coalesced_metal_load_ranges
            } else {
                &legacy_metal_load_ranges
            },
            &tensor_name_to_metal_output_index,
            &selected_page_slot_ids,
            &layer_plan,
        );
        let second_measurement = measure_variant(
            qwen3_5_model.runtime(),
            &aligned_expert_pack_path,
            &metal_output_tensors,
            if coalesced_runs_first {
                &legacy_metal_load_ranges
            } else {
                &coalesced_metal_load_ranges
            },
            &tensor_name_to_metal_output_index,
            &selected_page_slot_ids,
            &layer_plan,
        );
        let (legacy_measurement, coalesced_measurement) = if coalesced_runs_first {
            (second_measurement, first_measurement)
        } else {
            (first_measurement, second_measurement)
        };
        assert_projection_outputs_match(&legacy_measurement, &coalesced_measurement);
        legacy_measurements.push(legacy_measurement);
        coalesced_measurements.push(coalesced_measurement);
    }

    let legacy_average_measurement = average_measurement(&legacy_measurements);
    let coalesced_average_measurement = average_measurement(&coalesced_measurements);
    assert_eq!(
        legacy_average_measurement.requested_byte_count,
        coalesced_average_measurement.requested_byte_count,
        "both variants must transfer the identical logical expert payload"
    );
    assert_eq!(
        legacy_average_measurement.command_count,
        legacy_metal_load_ranges.len(),
        "native Metal I/O must encode every legacy descriptor as one command"
    );
    assert_eq!(
        coalesced_average_measurement.command_count,
        coalesced_metal_load_ranges.len(),
        "native Metal I/O must encode every coalesced descriptor as one command"
    );
    let final_memory_snapshot = qwen3_5_model
        .runtime()
        .memory_snapshot()
        .expect("the completed model-resident benchmark should report MLX memory");
    eprintln!(
        "[oq4e-10gb-ranges] status=success total_elapsed_seconds={:.3} configured_memory_limit_bytes={} model_load_active_bytes={} model_activation_active_bytes={} final_active_bytes={} final_peak_bytes={} selected_expert_count={} logical_payload_bytes={} legacy_command_count={} legacy_elapsed_ms={:.2} legacy_host_encoding_ns={} legacy_queue_elapsed_ns={} coalesced_command_count={} coalesced_elapsed_ms={:.2} coalesced_host_encoding_ns={} coalesced_queue_elapsed_ns={}",
        benchmark_started_at.elapsed().as_secs_f64(),
        CONFIGURED_MLX_MEMORY_LIMIT_BYTES,
        model_load_memory_snapshot.active_memory_bytes(),
        model_activation_memory_snapshot.active_memory_bytes(),
        final_memory_snapshot.active_memory_bytes(),
        final_memory_snapshot.peak_memory_bytes(),
        selected_expert_ids.len(),
        legacy_average_measurement.requested_byte_count,
        legacy_average_measurement.command_count,
        legacy_average_measurement.elapsed.as_secs_f64() * 1_000.0,
        legacy_average_measurement.host_encoding_elapsed_nanoseconds,
        legacy_average_measurement.queue_elapsed_nanoseconds,
        coalesced_average_measurement.command_count,
        coalesced_average_measurement.elapsed.as_secs_f64() * 1_000.0,
        coalesced_average_measurement.host_encoding_elapsed_nanoseconds,
        coalesced_average_measurement.queue_elapsed_nanoseconds,
    );
}

struct VariantMeasurement {
    projection_outputs: HashMap<&'static str, MlxArray>,
    metrics: MlxMetalExpertPackLoadMetrics,
    elapsed: Duration,
}

struct AverageVariantMeasurement {
    requested_byte_count: u64,
    command_count: usize,
    host_encoding_elapsed_nanoseconds: u64,
    queue_elapsed_nanoseconds: u64,
    elapsed: Duration,
}

#[allow(clippy::too_many_arguments)]
fn measure_variant(
    runtime: &MlxRuntime,
    aligned_expert_pack_path: &std::path::Path,
    metal_output_tensors: &[MlxMetalExpertPackOutputTensor],
    metal_load_ranges: &[MlxMetalExpertPackLoadRange],
    tensor_name_to_metal_output_index: &HashMap<String, usize>,
    selected_page_slot_ids: &[i32],
    layer_plan: &astronomical_model_serving::QuantizedExpertLayerPlan,
) -> VariantMeasurement {
    let (projection_outputs, metrics, elapsed) = measure_metal_io_projections(
        runtime,
        aligned_expert_pack_path,
        metal_output_tensors,
        metal_load_ranges,
        tensor_name_to_metal_output_index,
        selected_page_slot_ids,
        layer_plan,
    );
    VariantMeasurement {
        projection_outputs,
        metrics,
        elapsed,
    }
}

fn split_coalesced_ranges_into_legacy_ranges(
    coalesced_metal_load_ranges: &[MlxMetalExpertPackLoadRange],
    tensor_descriptors: &[astronomical_model_serving::AlignedExpertPackTensorDescriptor],
) -> Vec<MlxMetalExpertPackLoadRange> {
    let mut legacy_metal_load_ranges = Vec::new();
    for coalesced_load_range in coalesced_metal_load_ranges {
        let tensor_descriptor = tensor_descriptors
            .get(coalesced_load_range.output_tensor_index())
            .expect("every coalesced range should reference a prepared tensor descriptor");
        assert_eq!(
            coalesced_load_range.byte_count() % tensor_descriptor.bytes_per_expert,
            0,
            "a coalesced range should contain complete experts only"
        );
        for expert_byte_offset in
            (0..coalesced_load_range.byte_count()).step_by(tensor_descriptor.bytes_per_expert)
        {
            legacy_metal_load_ranges.push(MlxMetalExpertPackLoadRange::new(
                coalesced_load_range.output_tensor_index(),
                coalesced_load_range.output_tensor_offset_bytes() + expert_byte_offset,
                coalesced_load_range.source_file_offset_bytes()
                    + u64::try_from(expert_byte_offset)
                        .expect("the selected expert byte offset should fit u64"),
                tensor_descriptor.bytes_per_expert,
            ));
        }
    }
    legacy_metal_load_ranges
}

fn assert_projection_outputs_match(
    legacy_measurement: &VariantMeasurement,
    coalesced_measurement: &VariantMeasurement,
) {
    for projection_name in ["gate_proj", "up_proj", "down_proj"] {
        assert_eq!(
            legacy_measurement.projection_outputs[projection_name]
                .to_vec_f32()
                .expect("the legacy projection output should copy to the host for parity"),
            coalesced_measurement.projection_outputs[projection_name]
                .to_vec_f32()
                .expect("the coalesced projection output should copy to the host for parity"),
            "coalesced {projection_name} output must exactly match legacy per-expert I/O"
        );
    }
}

fn average_measurement(measurements: &[VariantMeasurement]) -> AverageVariantMeasurement {
    assert!(
        !measurements.is_empty(),
        "the benchmark should collect measurements"
    );
    let measurement_count =
        u64::try_from(measurements.len()).expect("the benchmark measurement count should fit u64");
    let first_measurement = &measurements[0];
    assert!(
        measurements.iter().all(|measurement| {
            measurement.metrics.requested_byte_count
                == first_measurement.metrics.requested_byte_count
                && measurement.metrics.command_count == first_measurement.metrics.command_count
        }),
        "every repeated measurement must request the same payload bytes and commands"
    );
    AverageVariantMeasurement {
        requested_byte_count: first_measurement.metrics.requested_byte_count,
        command_count: first_measurement.metrics.command_count,
        host_encoding_elapsed_nanoseconds: measurements
            .iter()
            .map(|measurement| measurement.metrics.host_encoding_elapsed_nanoseconds)
            .sum::<u64>()
            / measurement_count,
        queue_elapsed_nanoseconds: measurements
            .iter()
            .map(|measurement| measurement.metrics.queue_elapsed_nanoseconds)
            .sum::<u64>()
            / measurement_count,
        elapsed: Duration::from_nanos(
            measurements
                .iter()
                .map(|measurement| {
                    u64::try_from(measurement.elapsed.as_nanos())
                        .expect("each benchmark elapsed duration should fit u64 nanoseconds")
                })
                .sum::<u64>()
                / measurement_count,
        ),
    }
}
