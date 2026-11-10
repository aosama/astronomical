//! What the parity journeys share: the conversion OptiQ's reference
//! builder ran, the comparison of two artifacts, and the rank statistics a
//! sweep is judged by. Kept apart from the journeys themselves so the dense
//! and the mixture-of-experts journeys use one definition of each.

use std::collections::BTreeMap;
use std::path::Path;

use astronomical_model_serving::quantization_probe::load_qwen3_5_probe_model;
use astronomical_quantization::allocation::AllocationConstraints;
use astronomical_quantization::bit_width::{CandidateBitWidths, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::pipeline::{
    PipelineMethod, ProbeModelHandle, ProbeModelLoader, ProbeModelSweeper,
    QuantizationPipelineRequest, ReferenceMode,
};

use super::artifact_comparison::{json_differences, read_json_document, tensor_differences};
use super::support::{
    PREFILL_CHUNK_TOKENS, journey_memory_limits, python_reference_directory, source_model_directory,
};

/// The reference builder's defaults.
pub(super) const TARGET_BITS_PER_WEIGHT: f64 = 5.0;
pub(super) const CANDIDATE_BITS: [u32; 2] = [4, 8];
pub(super) const CALIBRATION_SEQUENCE_LENGTH_TOKENS: usize = 512;

/// The documents compared key by key. `base_model` records the path each
/// tool was handed, which is the caller's spelling, not the tool's output.
pub(super) const COMPARED_DOCUMENTS: [&str; 4] = [
    "config.json",
    "optiq/metadata.json",
    "optiq/sensitivity.json",
    "model.safetensors.index.json",
];
pub(super) const CALLER_SPELLED_KEYS: [&str; 1] = ["base_model"];

/// The folder that holds OptiQ's own documents and the prediction-head
/// sidecar.
pub(super) const OPTIQ_SUBDIRECTORY: &str = "optiq";

/// The conversion the reference builder ran, as a pipeline request.
pub(super) fn parity_request(
    output_directory: &Path,
    method: PipelineMethod,
    calibration_sequence_count: usize,
) -> QuantizationPipelineRequest {
    QuantizationPipelineRequest {
        source_directory: source_model_directory(),
        output_directory: output_directory.to_path_buf(),
        base_model_name: "acceptance/parity".to_owned(),
        method,
        reference_mode: ReferenceMode::Bfloat16,
        target_bits_per_weight: TARGET_BITS_PER_WEIGHT,
        candidate_bits: CANDIDATE_BITS.to_vec(),
        group_size: GroupSize::new(64).expect("valid"),
        calibration_sequence_count,
        calibration_sequence_length_tokens: CALIBRATION_SEQUENCE_LENGTH_TOKENS,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_policy_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    }
}

pub(super) fn parity_constraints() -> AllocationConstraints {
    AllocationConstraints {
        target_bits_per_weight: TARGET_BITS_PER_WEIGHT,
        candidate_bit_widths: Some(CandidateBitWidths::new(&CANDIDATE_BITS).expect("valid ladder")),
        group_size: GroupSize::new(64).expect("valid"),
        protect_boundary_blocks: true,
        boundary_block_count: 1,
        floor_components_per_block: 2,
        maximum_low_bit_block_run: 3,
    }
}

/// How many sequences the reference sweep forwarded.
pub(super) fn reference_sequence_count() -> usize {
    let reference_sequences = read_json_document(
        &python_reference_directory(),
        "parity/calibration_sequences.json",
    );
    reference_sequences["sequences"]
        .as_array()
        .expect("the sequences are an array")
        .len()
}

/// Tensor and document differences between a reference artifact and a
/// candidate, plus any file present on one side only.
pub(super) fn artifact_differences(
    reference_directory: &Path,
    candidate_directory: &Path,
) -> Vec<String> {
    let mut differences = tensor_differences(reference_directory, candidate_directory);
    // The multi-token-prediction head lives under `optiq/`, out of reach of
    // a root-level glob; compare it wherever either artifact has one.
    let reference_sidecars = reference_directory.join(OPTIQ_SUBDIRECTORY);
    let candidate_sidecars = candidate_directory.join(OPTIQ_SUBDIRECTORY);
    if reference_sidecars.is_dir() && candidate_sidecars.is_dir() {
        for difference in tensor_differences(&reference_sidecars, &candidate_sidecars) {
            differences.push(format!("{OPTIQ_SUBDIRECTORY}/{difference}"));
        }
    }
    for document in COMPARED_DOCUMENTS {
        let reference_document = read_json_document(reference_directory, document);
        let candidate_document = read_json_document(candidate_directory, document);
        for difference in json_differences(
            &reference_document,
            &candidate_document,
            &CALLER_SPELLED_KEYS,
        ) {
            differences.push(format!("{document}{difference}"));
        }
    }
    let reference_files = top_level_file_names(reference_directory);
    let candidate_files = top_level_file_names(candidate_directory);
    for file_name in reference_files.symmetric_difference(&candidate_files) {
        differences.push(format!("{file_name}: present in only one artifact"));
    }
    differences
}

pub(super) fn top_level_file_names(directory: &Path) -> std::collections::BTreeSet<String> {
    let mut file_names = std::collections::BTreeSet::new();
    for directory_entry in std::fs::read_dir(directory)
        .expect("the artifact is readable")
        .flatten()
    {
        file_names.insert(directory_entry.file_name().to_string_lossy().into_owned());
    }
    file_names
}

pub(super) fn report_differences(subject: &str, differences: &[String]) {
    eprintln!("[parity] {subject}: {} differences", differences.len());
    for difference in differences.iter().take(20) {
        eprintln!("[parity]   {difference}");
    }
}

/// The share of modules both artifacts write at the same width.
pub(super) fn bit_map_agreement(reference_directory: &Path, candidate_directory: &Path) -> f64 {
    let reference_metadata = read_json_document(reference_directory, "optiq/metadata.json");
    let candidate_metadata = read_json_document(candidate_directory, "optiq/metadata.json");
    let reference_per_layer = reference_metadata["per_layer"]
        .as_object()
        .expect("an object");
    let mut agreeing_module_count = 0_usize;
    for (module_name, reference_entry) in reference_per_layer {
        if candidate_metadata["per_layer"][module_name]["bits"] == reference_entry["bits"] {
            agreeing_module_count += 1;
        }
    }
    agreeing_module_count as f64 / reference_per_layer.len() as f64
}

/// Spearman rank correlation of the two sweeps' KL values at one width.
pub(super) fn rank_correlation_at(
    reference_sensitivity: &serde_json::Value,
    measured_sensitivity: &serde_json::Value,
    bits: u32,
) -> f64 {
    let reference_divergences = divergence_by_module(reference_sensitivity, bits);
    let measured_divergences = divergence_by_module(measured_sensitivity, bits);
    let mut reference_values = Vec::new();
    let mut measured_values = Vec::new();
    for (module_name, reference_divergence) in &reference_divergences {
        reference_values.push(*reference_divergence);
        measured_values.push(measured_divergences[module_name]);
    }
    pearson_correlation(&ranks_of(&reference_values), &ranks_of(&measured_values))
}

/// Spearman rank correlation of the two sweeps' gain from one width to a
/// higher one, `kl(lower) - kl(higher)` per module.
pub(super) fn gain_rank_correlation(
    reference_sensitivity: &serde_json::Value,
    measured_sensitivity: &serde_json::Value,
    lower_bits: u32,
    higher_bits: u32,
) -> f64 {
    let reference_lower = divergence_by_module(reference_sensitivity, lower_bits);
    let reference_higher = divergence_by_module(reference_sensitivity, higher_bits);
    let measured_lower = divergence_by_module(measured_sensitivity, lower_bits);
    let measured_higher = divergence_by_module(measured_sensitivity, higher_bits);
    let mut reference_gains = Vec::new();
    let mut measured_gains = Vec::new();
    for (module_name, reference_lower_divergence) in &reference_lower {
        reference_gains.push(reference_lower_divergence - reference_higher[module_name]);
        measured_gains.push(measured_lower[module_name] - measured_higher[module_name]);
    }
    pearson_correlation(&ranks_of(&reference_gains), &ranks_of(&measured_gains))
}

pub(super) fn divergence_by_module(
    sensitivity_document: &serde_json::Value,
    bits: u32,
) -> BTreeMap<String, f64> {
    let mut divergence_by_module = BTreeMap::new();
    for layer in sensitivity_document["layers"]
        .as_array()
        .expect("a layer list")
    {
        let module_name = layer["layer_name"]
            .as_str()
            .expect("a module name")
            .to_owned();
        let divergence = layer["sensitivities"][bits.to_string()]
            .as_f64()
            .expect("a divergence");
        divergence_by_module.insert(module_name, divergence);
    }
    divergence_by_module
}

/// Each value's position in ascending order.
pub(super) fn ranks_of(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    for (rank, value_index) in order.into_iter().enumerate() {
        ranks[value_index] = rank as f64;
    }
    ranks
}

pub(super) fn pearson_correlation(left_values: &[f64], right_values: &[f64]) -> f64 {
    let value_count = left_values.len() as f64;
    let left_mean = left_values.iter().sum::<f64>() / value_count;
    let right_mean = right_values.iter().sum::<f64>() / value_count;
    let (mut covariance, mut left_variance, mut right_variance) = (0.0, 0.0, 0.0);
    for (left_value, right_value) in left_values.iter().zip(right_values) {
        covariance += (left_value - left_mean) * (right_value - right_mean);
        left_variance += (left_value - left_mean).powi(2);
        right_variance += (right_value - right_mean).powi(2);
    }
    covariance / (left_variance.sqrt() * right_variance.sqrt())
}

/// Loads the resident Qwen3.5 probe under the journey's memory limits.
pub(super) struct Qwen3_5ProbeLoader;

impl ProbeModelLoader for Qwen3_5ProbeLoader {
    fn load_probe_model(
        &mut self,
        probe_directory: &Path,
        streaming_source_directory: Option<&Path>,
    ) -> Result<Box<dyn ProbeModelHandle>, QuantizationError> {
        let probe_model = load_qwen3_5_probe_model(
            probe_directory,
            streaming_source_directory,
            journey_memory_limits(),
            PREFILL_CHUNK_TOKENS,
        )?;
        Ok(Box::new(ProbeModelSweeper(probe_model)))
    }
}
