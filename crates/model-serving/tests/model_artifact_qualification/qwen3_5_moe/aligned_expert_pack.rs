use std::{
    collections::HashMap,
    fs::File,
    os::unix::fs::FileExt,
    time::{Duration, Instant},
};

use astronomical_ipc_protocol::ExpertStorageFormat;
use astronomical_model_serving::{
    AlignedExpertPackPreparer, ExpertPager, QuantizationMode, Qwen3_5ArtifactValidator,
    build_aligned_expert_pack_metal_io_descriptors, build_quantized_expert_layer_plan,
    build_quantized_expert_page_manifest_from_plan,
};
use astronomical_runtime_integration::MlxRuntime;
use tokio::time::{MissedTickBehavior, interval, sleep};

use astronomical_model_serving::{
    AlignedExpertPackBuildRequest, AlignedExpertPackTensorDescriptor, build_aligned_expert_pack,
    validate_aligned_expert_pack_header,
};

use super::aligned_expert_pack_projection::{
    measure_bounded_reader_projections, measure_metal_io_projections, selected_expert_ids,
};

const ALIGNED_EXPERT_PACK_TEST_TIMEOUT: Duration = Duration::from_secs(120);

#[test]
#[ignore = "inspects the installed model without creating aligned expert packs"]
fn should_inspect_the_downloaded_model_before_explicit_pack_preparation() {
    let model_directory = crate::common::configured_model_directory_by_id("Ornith-1.0-35B-8bit")
        .expect("the uniform Ornith 8-bit checkpoint should be installed");
    let preparer = AlignedExpertPackPreparer::for_model_directory(&model_directory)
        .expect("the downloaded model should support aligned expert-pack preparation");

    let preparation_inspection = preparer
        .inspect()
        .expect("aligned expert-pack preparation should inspect without mutation");

    assert_eq!(preparation_inspection.total_layer_count, 40);
    assert!(preparation_inspection.total_pack_byte_count > 0);
    assert!(preparation_inspection.available_byte_count > 0);
}

#[tokio::test]
#[ignore = "activates all explicitly prepared layers through production ExpertPager discovery"]
async fn should_activate_the_complete_prepared_revision_in_the_production_expert_pager() {
    let model_directory = crate::common::configured_model_directory_by_id("Ornith-1.0-35B-8bit")
        .expect("the uniform Ornith 8-bit checkpoint should be installed");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the prepared model should validate");
    let config = validated_artifact.config().clone();
    let configured_mlx_memory_cap_bytes =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits()
            .await
            .active_memory_limit_bytes();
    let weight_map = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect::<HashMap<_, _>>();
    let expert_pager = ExpertPager::new(
        model_directory,
        validated_artifact.model_id(),
        validated_artifact.revision(),
        &weight_map,
        &config,
        configured_mlx_memory_cap_bytes,
        false,
    )
    .expect("production ExpertPager should construct with the prepared revision");

    assert_eq!(expert_pager.aligned_expert_pack_layer_count(), 40);
    assert_eq!(
        expert_pager.expert_storage_format(),
        ExpertStorageFormat::AstronomicalAligned
    );
}

#[tokio::test]
#[ignore = "activates prepared decoder packs while the optional MTP expert layer remains in source shards"]
async fn should_report_astronomical_optimized_storage_with_a_source_backed_mtp_expert_layer() {
    require_aligned_expert_pack_completion(async {
        let model_directory =
            crate::common::configured_model_directory_by_id("Qwen3.6-35B-A3B-oQ4e-mtp")
                .expect("the Qwen MTP checkpoint should be installed");
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the prepared Qwen MTP model should validate");
        let config = validated_artifact.config().clone();
        let configured_mlx_memory_cap_bytes =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits()
                .await
                .active_memory_limit_bytes();
        let weight_map = validated_artifact
            .shard_index()
            .language_tensor_name_to_shard_file_name()
            .iter()
            .chain(
                validated_artifact
                    .shard_index()
                    .mtp_tensor_name_to_shard_file_name(),
            )
            .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
            .collect::<HashMap<_, _>>();
        let expert_pager = ExpertPager::new(
            model_directory,
            validated_artifact.model_id(),
            validated_artifact.revision(),
            &weight_map,
            &config,
            configured_mlx_memory_cap_bytes,
            true,
        )
        .expect("the prepared Qwen MTP pager should construct");

        assert_eq!(expert_pager.layer_count(), 41);
        assert_eq!(expert_pager.aligned_expert_pack_layer_count(), 40);
        assert_eq!(
            expert_pager.expert_storage_format(),
            ExpertStorageFormat::AstronomicalAligned
        );
    })
    .await;
}
const PACK_COPY_SCRATCH_BYTES: usize = 64 * 1024;
const ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN: usize = 8;
const PROMPT_PROCESSING_PREFILL_CHUNCK_TOKENS: usize = 2_048;

#[derive(Clone, Copy)]
struct ExpertDataPlaneScenario {
    scenario_name: &'static str,
    routed_token_count: usize,
    selected_expert_count: usize,
}

impl ExpertDataPlaneScenario {
    const fn generation() -> Self {
        Self {
            scenario_name: "generation",
            routed_token_count: 1,
            selected_expert_count: ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN,
        }
    }

    const fn prompt_processing() -> Self {
        Self {
            scenario_name: "prompt_processing",
            routed_token_count: PROMPT_PROCESSING_PREFILL_CHUNCK_TOKENS,
            selected_expert_count: 256,
        }
    }

    fn routed_page_slot_ids(self) -> Vec<i32> {
        let routed_assignment_count = self
            .routed_token_count
            .checked_mul(ROUTED_EXPERT_ASSIGNMENTS_PER_TOKEN)
            .expect("routed assignment count should fit usize");
        (0..routed_assignment_count)
            .map(|routed_assignment_position| {
                i32::try_from(routed_assignment_position % self.selected_expert_count)
                    .expect("selected expert page slot should fit i32")
            })
            .collect()
    }
}

#[tokio::test]
#[ignore = "measures one generation token through bounded and direct Metal gather-QMM paths"]
async fn should_measure_one_layer_generation_expert_data_plane() {
    require_aligned_expert_pack_completion(run_one_layer_expert_data_plane(
        ExpertDataPlaneScenario::generation(),
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures one prompt-processing chunk through bounded and direct Metal gather-QMM paths"]
async fn should_measure_one_layer_prompt_processing_expert_data_plane() {
    require_aligned_expert_pack_completion(run_one_layer_expert_data_plane(
        ExpertDataPlaneScenario::prompt_processing(),
    ))
    .await;
}

async fn run_one_layer_expert_data_plane(data_plane_scenario: ExpertDataPlaneScenario) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_model_directory_by_id("Ornith-1.0-35B-8bit")
        .expect("the uniform Ornith 8-bit checkpoint should be installed");
    eprintln!("[aligned-expert-pack] status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the configured Ornith artifact should validate");
    let qwen3_5_config = validated_artifact.config().clone();
    let language_tensor_name_to_shard_file_name = validated_artifact
        .shard_index()
        .language_tensor_name_to_shard_file_name()
        .iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name.clone(), shard_file_name.clone()))
        .collect::<HashMap<_, _>>();
    let layer_prefix = "language_model.model.layers.0.mlp";
    let layer_plan = build_quantized_expert_layer_plan(
        &model_directory,
        &language_tensor_name_to_shard_file_name,
        layer_prefix,
        &qwen3_5_config,
        QuantizationMode::Affine,
    )
    .expect("the first Ornith layer should have one validated affine expert plan");
    assert_eq!(layer_plan.tensor_sources.len(), 9);
    assert_eq!(layer_plan.quantization_bits, 8);
    assert_eq!(layer_plan.quantization_group_size, 64);
    assert_eq!(layer_plan.quantization_mode, QuantizationMode::Affine);

    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let aligned_expert_pack_path = temporary_directory.path().join("layer-0.expert-pack");
    eprintln!("[aligned-expert-pack] status=progress phase=pack_build");
    let pack_build_started_at = Instant::now();
    let aligned_expert_pack_header = build_aligned_expert_pack(
        &aligned_expert_pack_path,
        &AlignedExpertPackBuildRequest {
            model_id: validated_artifact.model_id(),
            model_revision: validated_artifact.revision(),
            layer_index: 0,
            layer_plan: &layer_plan,
        },
    )
    .expect("the first Ornith layer should pack with bounded scratch memory");
    validate_aligned_expert_pack_header(
        &aligned_expert_pack_path,
        &aligned_expert_pack_header,
        &layer_plan,
        validated_artifact.model_id(),
        validated_artifact.revision(),
        0,
    )
    .expect("the reopened pack header should match the validated model layer");
    let pack_build_elapsed = pack_build_started_at.elapsed();
    eprintln!(
        "[aligned-expert-pack] status=progress phase=pack_validated model_id={} quantization_bits={} quantization_group_size={} pack_bytes={} pack_build_ms={:.2}",
        validated_artifact.model_id(),
        layer_plan.quantization_bits,
        layer_plan.quantization_group_size,
        aligned_expert_pack_header.expected_pack_byte_count,
        pack_build_elapsed.as_secs_f64() * 1_000.0,
    );
    for tensor_descriptor in &aligned_expert_pack_header.tensor_descriptors {
        assert_packed_tensor_matches_source(
            &aligned_expert_pack_path,
            tensor_descriptor,
            &layer_plan,
        );
    }

    let selected_expert_ids = selected_expert_ids(
        layer_plan.expert_capacity,
        data_plane_scenario.selected_expert_count,
    );
    let reference_page_manifest =
        build_quantized_expert_page_manifest_from_plan(&layer_plan, &selected_expert_ids)
            .expect("the selected Ornith experts should form a bounded reader manifest");
    eprintln!("[aligned-expert-pack] status=progress phase=runtime_init");
    let runtime = MlxRuntime::initialize(
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await,
    )
    .expect("the direct MLX runtime should initialize");
    let selected_page_slot_ids = data_plane_scenario.routed_page_slot_ids();
    let (metal_output_tensors, metal_load_ranges) = build_aligned_expert_pack_metal_io_descriptors(
        &aligned_expert_pack_header.tensor_descriptors,
        &layer_plan,
        &selected_expert_ids,
    )
    .expect("the selected experts should produce Metal I/O descriptors");
    let tensor_name_to_metal_output_index =
        tensor_name_to_metal_output_indices(&aligned_expert_pack_header.tensor_descriptors);
    eprintln!("[aligned-expert-pack] status=progress phase=measurement_warmup");
    let _bounded_reader_warmup = measure_bounded_reader_projections(
        &runtime,
        &reference_page_manifest,
        &selected_page_slot_ids,
        &layer_plan,
    );
    let _metal_io_warmup = measure_metal_io_projections(
        &runtime,
        &aligned_expert_pack_path,
        &metal_output_tensors,
        &metal_load_ranges,
        &tensor_name_to_metal_output_index,
        &selected_page_slot_ids,
        &layer_plan,
    );
    eprintln!("[aligned-expert-pack] status=progress phase=measurement");
    let (bounded_reader_projection_outputs, bounded_reader_elapsed) =
        measure_bounded_reader_projections(
            &runtime,
            &reference_page_manifest,
            &selected_page_slot_ids,
            &layer_plan,
        );
    let (metal_projection_outputs, metal_io_metrics, metal_io_elapsed) =
        measure_metal_io_projections(
            &runtime,
            &aligned_expert_pack_path,
            &metal_output_tensors,
            &metal_load_ranges,
            &tensor_name_to_metal_output_index,
            &selected_page_slot_ids,
            &layer_plan,
        );

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
    eprintln!(
        "[aligned-expert-pack] status=success scenario={} routed_token_count={} routed_assignment_count={} selected_expert_count={} bounded_reader_elapsed_ms={:.2} metal_io_elapsed_ms={:.2} requested_bytes={} command_count={} host_encoding_ns={} queue_elapsed_ns={}",
        data_plane_scenario.scenario_name,
        data_plane_scenario.routed_token_count,
        selected_page_slot_ids.len(),
        selected_expert_ids.len(),
        bounded_reader_elapsed.as_secs_f64() * 1_000.0,
        metal_io_elapsed.as_secs_f64() * 1_000.0,
        metal_io_metrics.requested_byte_count,
        metal_io_metrics.command_count,
        metal_io_metrics.host_encoding_elapsed_nanoseconds,
        metal_io_metrics.queue_elapsed_nanoseconds,
    );
}

fn tensor_name_to_metal_output_indices(
    tensor_descriptors: &[AlignedExpertPackTensorDescriptor],
) -> HashMap<String, usize> {
    let mut tensor_name_to_metal_output_index = HashMap::new();
    for (output_tensor_index, tensor_descriptor) in tensor_descriptors.iter().enumerate() {
        tensor_name_to_metal_output_index
            .insert(short_tensor_name(tensor_descriptor), output_tensor_index);
    }
    tensor_name_to_metal_output_index
}

fn assert_packed_tensor_matches_source(
    aligned_expert_pack_path: &std::path::Path,
    tensor_descriptor: &AlignedExpertPackTensorDescriptor,
    layer_plan: &astronomical_model_serving::QuantizedExpertLayerPlan,
) {
    let tensor_source = layer_plan
        .tensor_sources
        .iter()
        .find(|tensor_source| tensor_source.tensor_name == tensor_descriptor.tensor_name)
        .expect("every packed descriptor should reference a layer-plan source");
    let source_file = File::open(&tensor_source.source_file)
        .expect("the validated source shard should remain readable");
    let aligned_expert_pack_file = File::open(aligned_expert_pack_path)
        .expect("the freshly written aligned expert pack should remain readable");
    let mut compared_byte_count = 0_usize;
    let mut source_scratch_bytes = vec![0_u8; PACK_COPY_SCRATCH_BYTES];
    let mut packed_scratch_bytes = vec![0_u8; PACK_COPY_SCRATCH_BYTES];
    while compared_byte_count < tensor_descriptor.logical_byte_count {
        let next_compare_byte_count = (tensor_descriptor.logical_byte_count - compared_byte_count)
            .min(PACK_COPY_SCRATCH_BYTES);
        source_file
            .read_exact_at(
                &mut source_scratch_bytes[..next_compare_byte_count],
                tensor_descriptor.source_payload_offset_bytes
                    + u64::try_from(compared_byte_count).expect("comparison offset should fit u64"),
            )
            .expect("the validated source tensor range should be readable");
        aligned_expert_pack_file
            .read_exact_at(
                &mut packed_scratch_bytes[..next_compare_byte_count],
                tensor_descriptor.pack_segment_offset_bytes
                    + u64::try_from(compared_byte_count).expect("comparison offset should fit u64"),
            )
            .expect("the aligned expert pack tensor range should be readable");
        assert_eq!(
            &source_scratch_bytes[..next_compare_byte_count],
            &packed_scratch_bytes[..next_compare_byte_count],
            "{} must preserve every source byte",
            tensor_descriptor.tensor_name
        );
        compared_byte_count += next_compare_byte_count;
    }
}

fn short_tensor_name(tensor_descriptor: &AlignedExpertPackTensorDescriptor) -> String {
    format!(
        "{}.{}",
        tensor_descriptor.projection_name, tensor_descriptor.parameter_name
    )
}

async fn require_aligned_expert_pack_completion(
    test_future: impl std::future::Future<Output = ()>,
) {
    let started_at = Instant::now();
    let timeout_deadline = sleep(ALIGNED_EXPERT_PACK_TEST_TIMEOUT);
    let mut progress_interval = interval(Duration::from_secs(10));
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    tokio::pin!(test_future);
    tokio::pin!(timeout_deadline);
    progress_interval.tick().await;
    loop {
        tokio::select! {
            () = &mut test_future => return,
            () = &mut timeout_deadline => panic!("the aligned expert-pack model-artifact benchmark exceeded {} seconds", ALIGNED_EXPERT_PACK_TEST_TIMEOUT.as_secs()),
            _ = progress_interval.tick() => eprintln!(
                "[aligned-expert-pack] status=running elapsed_seconds={:.0} ETA<={:.0}",
                started_at.elapsed().as_secs_f64(),
                ALIGNED_EXPERT_PACK_TEST_TIMEOUT.saturating_sub(started_at.elapsed()).as_secs_f64()
            ),
        }
    }
}
