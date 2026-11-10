//! Parity with OptiQ, the Python tool this crate ports, on a real base.
//!
//! The journeys take a bfloat16 source as `model_directory` and, as
//! `reference_artifact_path`, the directory
//! `crates/quantization/tests/fixtures/build_optiq_reference_artifacts.py`
//! wrote from the same source: OptiQ's `static_mixed/` and `optiq_mixed/`
//! artifacts and the calibration sequences its sweep forwarded.
//!
//! What can be identical is asserted identical: the structural conversion
//! byte for byte, the calibration sequences token for token, and the bit map
//! the Rust allocator derives from OptiQ's own sweep. The sweep itself runs
//! through a different engine on each side, so its KL values cannot match to
//! the last digit; that journey asserts the two rankings agree instead.

use std::path::Path;

use astronomical_model_serving::quantization_probe::Qwen3_5KlDivergenceEngine;
use astronomical_quantization::artifact::OptiQSensitivityDocument;
use astronomical_quantization::calibration::mix::DEFAULT_SHUFFLE_SEED;
use astronomical_quantization::calibration::{CalibrationMix, HuggingFaceCalibrationTokenizer};
use astronomical_quantization::evaluation::{
    KlEvaluationRequest, bundled_evaluation_prompts, evaluate_kl_divergence,
    tokenized_evaluation_prompts,
};
use astronomical_quantization::pipeline::probe_source_view::loadable_model_directory;
use astronomical_quantization::pipeline::{
    PipelineMethod, plan_from_sensitivity_table, run_quantization_pipeline,
};
use serial_test::serial;

use super::artifact_comparison::{
    json_differences, read_json_document, read_stored_tensors, tensor_differences,
};
use super::parity_support::{
    CALIBRATION_SEQUENCE_LENGTH_TOKENS, CANDIDATE_BITS, Qwen3_5ProbeLoader, artifact_differences,
    bit_map_agreement, gain_rank_correlation, parity_constraints, parity_request,
    rank_correlation_at, reference_sequence_count, report_differences,
};
use super::support::{
    PREFILL_CHUNK_TOKENS, StandardErrorProgress, fresh_output_directory, journey_memory_limits,
    journey_runtime, python_reference_directory, source_model_directory,
};

/// The reference builder's low-bit ladder, which puts modules at 2 and 3
/// bits, the widths encoded by range search.
const LOW_BIT_CANDIDATE_BITS: [u32; 4] = [2, 3, 4, 8];
const LOW_BIT_TARGET_BITS_PER_WEIGHT: f64 = 3.5;
const RANGE_SEARCH_MAXIMUM_BITS: u64 = 3;
const BFLOAT16_BYTES: usize = 2;

/// The range search keeps a candidate only when it is strictly better, so
/// the last bit of the error decides every near-tie, and OptiQ computes that
/// error with a Metal kernel that rounds differently from build to build.
/// Measured on Qwen3.5 0.8B: 0.008% of searched groups. Everything else in
/// the artifact, every module at 4 bits and above included, is identical.
const MAXIMUM_NEAR_TIE_GROUP_SHARE: f64 = 0.001;

/// The two evaluators run the same prompts through different engines.
/// Measured on the original Qwen3.5 0.8B release: 0.1029 here against
/// 0.1077 for the mean, 0.0350 against 0.0358 for the median, and 0.3451
/// against 0.3506 for the 95th percentile, each within five percent.
const MAXIMUM_EVALUATION_RELATIVE_DIFFERENCE: f64 = 0.10;

/// The sweep checkpoint both tools keep beside the artifact directory.
const SWEEP_CHECKPOINT_FILE_NAME: &str = "sensitivity_checkpoint.json";

/// Two engines measure the same quantity through different arithmetic, so
/// the rankings must agree closely but cannot agree exactly. Two rankings
/// are held to the threshold: the divergence at the ladder's lowest width,
/// and the gain from the lowest width to the highest, which is the quantity
/// the allocator spends its budget on.
///
/// The divergence at the highest width is reported and not asserted. On a
/// real bfloat16 base an 8-bit probe moves the output so little that
/// OptiQ's measurement sits on its own noise floor (its logits are
/// bfloat16: median 9.8e-4 on Qwen3.5 0.8B), while this engine computes the
/// head in float32 and reads about 45 times lower (2.1e-5). The two
/// columns then rank mostly noise against signal (0.63), and it does not
/// matter: the gain still correlates at 0.993.
///
/// Measured on the original Qwen3.5 0.8B release with four sequences: 0.961
/// at 4 bits, 0.993 on the gain, 182 of 186 modules at the same width, and
/// the same achieved bits per weight.
const MINIMUM_RANK_CORRELATION: f64 = 0.95;
const MINIMUM_BIT_MAP_AGREEMENT: f64 = 0.85;

#[test]
#[serial]
#[ignore = "needs a bfloat16 model_directory and the reference_artifact_path the reference builder wrote from it"]
fn should_write_the_python_static_artifact_byte_for_byte() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("parity-static");
    let request = parity_request(&output_directory, PipelineMethod::StaticStructural, 0);

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Qwen3_5ProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the static conversion should succeed");

    let reference_directory = python_reference_directory().join("static_mixed");
    let differences = artifact_differences(&reference_directory, &outcome.artifact_directory);
    report_differences("static artifact", &differences);
    assert!(differences.is_empty(), "{} differences", differences.len());
    let _ = std::fs::remove_dir_all(&output_directory);
}

#[test]
#[serial]
#[ignore = "needs a bfloat16 model_directory and the reference_artifact_path the reference builder wrote from it"]
fn should_write_the_python_measured_artifact_byte_for_byte_from_the_python_sweep() {
    // The sweep checkpoint is the one input two engines cannot reproduce to
    // the last digit, so this journey resumes from OptiQ's: every module is
    // already measured, nothing is re-probed, and everything downstream of
    // the measurement (allocation, encoding, layout, documents) must then
    // match OptiQ's measured artifact exactly.
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("parity-resumed");
    std::fs::copy(
        python_reference_directory().join(SWEEP_CHECKPOINT_FILE_NAME),
        output_directory.join(SWEEP_CHECKPOINT_FILE_NAME),
    )
    .expect("the Python sweep checkpoint should copy");
    let request = parity_request(
        &output_directory,
        PipelineMethod::ExactKlSensitivity,
        reference_sequence_count(),
    );

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Qwen3_5ProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the resumed conversion should succeed");

    let reference_directory = python_reference_directory().join("optiq_mixed");
    let differences = artifact_differences(&reference_directory, &outcome.artifact_directory);
    report_differences("measured artifact from the Python sweep", &differences);
    assert!(differences.is_empty(), "{} differences", differences.len());
    let _ = std::fs::remove_dir_all(&output_directory);
}

#[test]
#[ignore = "needs the reference_artifact_path the reference builder wrote"]
fn should_reproduce_the_python_bit_map_from_the_python_sensitivity_document() {
    let reference_directory = python_reference_directory().join("optiq_mixed");
    let sensitivity_document =
        OptiQSensitivityDocument::read(&reference_directory.join("optiq/sensitivity.json"))
            .expect("the Python sweep should parse");
    let reference_metadata = read_json_document(&reference_directory, "optiq/metadata.json");
    let sensitivity_table = sensitivity_document
        .to_table()
        .expect("the Python sweep should translate");

    let allocation = plan_from_sensitivity_table(&sensitivity_table, &parity_constraints())
        .expect("the allocation should succeed");

    let mut mismatched_modules = Vec::new();
    for decision in &allocation.decisions {
        let reference_bits =
            reference_metadata["per_layer"][&decision.module_name]["bits"].as_u64();
        if reference_bits != Some(u64::from(decision.bit_width.bits())) {
            mismatched_modules.push(decision.module_name.clone());
        }
    }
    let reference_module_count = reference_metadata["per_layer"]
        .as_object()
        .expect("per_layer is an object")
        .len();
    report_differences("bit map from the Python sweep", &mismatched_modules);
    assert_eq!(allocation.decisions.len(), reference_module_count);
    assert!(mismatched_modules.is_empty());
    let reference_achieved = reference_metadata["achieved_bpw"]
        .as_f64()
        .expect("a number");
    assert!((allocation.achieved_bits_per_weight - reference_achieved).abs() < 1e-9);
}

#[test]
#[ignore = "needs a model_directory with its tokenizer and the reference_artifact_path the reference builder wrote"]
fn should_cut_the_same_calibration_sequences_as_the_python_reference() {
    let reference_sequences = read_json_document(
        &python_reference_directory(),
        "parity/calibration_sequences.json",
    );
    let reference_sequences = reference_sequences["sequences"]
        .as_array()
        .expect("the sequences are an array");
    let tokenizer =
        HuggingFaceCalibrationTokenizer::from_model_directory(&source_model_directory())
            .expect("the source tokenizer should load");

    let calibration_mix = CalibrationMix::bundled(
        &tokenizer,
        (2 * reference_sequences.len()).max(16),
        CALIBRATION_SEQUENCE_LENGTH_TOKENS,
        DEFAULT_SHUFFLE_SEED,
        &mut StandardErrorProgress,
    )
    .expect("the bundled mix should tokenize");

    assert!(!reference_sequences.is_empty());
    for (sequence_index, reference_sequence) in reference_sequences.iter().enumerate() {
        let mut reference_token_ids = Vec::new();
        for token_id in reference_sequence.as_array().expect("a token list") {
            reference_token_ids
                .push(u32::try_from(token_id.as_u64().expect("a token id")).expect("fits"));
        }
        assert_eq!(
            calibration_mix.samples()[sequence_index].token_ids,
            reference_token_ids,
            "calibration sequence {sequence_index} differs from the Python reference"
        );
    }
}

#[test]
#[serial]
#[ignore = "needs a bfloat16 model_directory and the reference_artifact_path the reference builder wrote from it"]
fn should_rank_modules_like_the_python_sweep_and_land_on_a_close_bit_map() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("parity-measured");
    let reference_directory = python_reference_directory().join("optiq_mixed");
    let reference_sensitivity = read_json_document(&reference_directory, "optiq/sensitivity.json");
    let sequence_count = reference_sequence_count();
    let request = parity_request(
        &output_directory,
        PipelineMethod::ExactKlSensitivity,
        sequence_count,
    );

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Qwen3_5ProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the measured conversion should succeed");

    let measured_sensitivity =
        read_json_document(&outcome.artifact_directory, "optiq/sensitivity.json");
    let [lowest_bits, highest_bits] = CANDIDATE_BITS;
    let lowest_width_correlation =
        rank_correlation_at(&reference_sensitivity, &measured_sensitivity, lowest_bits);
    let highest_width_correlation =
        rank_correlation_at(&reference_sensitivity, &measured_sensitivity, highest_bits);
    let gain_correlation = gain_rank_correlation(
        &reference_sensitivity,
        &measured_sensitivity,
        lowest_bits,
        highest_bits,
    );
    eprintln!(
        "[parity] rank correlation at {lowest_bits} bits: {lowest_width_correlation:.4}; of the gain to {highest_bits} bits: {gain_correlation:.4}; at {highest_bits} bits (reported only): {highest_width_correlation:.4}"
    );
    assert!(lowest_width_correlation >= MINIMUM_RANK_CORRELATION);
    assert!(gain_correlation >= MINIMUM_RANK_CORRELATION);
    let agreement = bit_map_agreement(&reference_directory, &outcome.artifact_directory);
    eprintln!(
        "[parity] modules at the same width: {:.1}%",
        100.0 * agreement
    );
    assert!(agreement >= MINIMUM_BIT_MAP_AGREEMENT);
    let _ = std::fs::remove_dir_all(&output_directory);
}

#[test]
#[serial]
#[ignore = "needs a bfloat16 model_directory and the reference_artifact_path the reference builder wrote from it"]
fn should_match_the_python_low_bit_artifact_except_in_near_tie_groups() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("parity-low-bit");
    let mut request = parity_request(&output_directory, PipelineMethod::StaticStructural, 0);
    request.candidate_bits = LOW_BIT_CANDIDATE_BITS.to_vec();
    request.target_bits_per_weight = LOW_BIT_TARGET_BITS_PER_WEIGHT;

    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut Qwen3_5ProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the low-bit conversion should succeed");

    let reference_directory = python_reference_directory().join("low_bit/static_mixed");
    let reference_metadata = read_json_document(&reference_directory, "optiq/metadata.json");
    let candidate_metadata = read_json_document(&outcome.artifact_directory, "optiq/metadata.json");
    let bit_map_differences = json_differences(
        &reference_metadata["per_layer"],
        &candidate_metadata["per_layer"],
        &[],
    );
    let group_tally = searched_group_tally(&reference_directory, &outcome.artifact_directory);
    let differing_share = group_tally.differing_group_count as f64 / group_tally.group_count as f64;
    eprintln!(
        "[parity] range-searched groups: {} of {} differ ({:.4}%); differing tensors outside searched modules: {}",
        group_tally.differing_group_count,
        group_tally.group_count,
        100.0 * differing_share,
        group_tally.unsearched_differences.len()
    );
    assert!(bit_map_differences.is_empty());
    assert!(group_tally.group_count > 0);
    assert!(group_tally.unsearched_differences.is_empty());
    assert!(differing_share <= MAXIMUM_NEAR_TIE_GROUP_SHARE);
    let _ = std::fs::remove_dir_all(&output_directory);
}

/// How the two artifacts' range-searched modules compare, group by group.
struct SearchedGroupTally {
    group_count: usize,
    differing_group_count: usize,
    /// Tensors that differ although no range search touched them.
    unsearched_differences: Vec<String>,
}

/// Counts differing groups (one bfloat16 scale each) in modules at or below
/// the range search's widest width, and collects every differing tensor
/// that is not one of those modules' three tensors.
fn searched_group_tally(
    reference_directory: &Path,
    candidate_directory: &Path,
) -> SearchedGroupTally {
    let reference_metadata = read_json_document(reference_directory, "optiq/metadata.json");
    let reference_tensors = read_stored_tensors(reference_directory);
    let candidate_tensors = read_stored_tensors(candidate_directory);
    let mut tally = SearchedGroupTally {
        group_count: 0,
        differing_group_count: 0,
        unsearched_differences: Vec::new(),
    };
    let mut searched_tensor_names = std::collections::BTreeSet::new();
    for (module_name, per_layer_entry) in reference_metadata["per_layer"]
        .as_object()
        .expect("an object")
    {
        if per_layer_entry["bits"].as_u64().expect("a width") > RANGE_SEARCH_MAXIMUM_BITS {
            continue;
        }
        for tensor_suffix in ["weight", "scales", "biases"] {
            searched_tensor_names.insert(format!("{module_name}.{tensor_suffix}"));
        }
        let scales_name = format!("{module_name}.scales");
        let reference_scales = reference_tensors[&scales_name]
            .bytes
            .chunks_exact(BFLOAT16_BYTES);
        let candidate_scales = candidate_tensors[&scales_name]
            .bytes
            .chunks_exact(BFLOAT16_BYTES);
        for (reference_scale, candidate_scale) in reference_scales.zip(candidate_scales) {
            tally.group_count += 1;
            if reference_scale != candidate_scale {
                tally.differing_group_count += 1;
            }
        }
    }
    for difference in tensor_differences(reference_directory, candidate_directory) {
        let tensor_name = difference.split(':').next().unwrap_or_default();
        if !searched_tensor_names.contains(tensor_name) {
            tally.unsearched_differences.push(difference);
        }
    }
    tally
}

#[test]
#[serial]
#[ignore = "needs a bfloat16 model_directory and the reference_artifact_path the reference builder wrote from it"]
fn should_evaluate_the_python_artifact_like_the_python_evaluator() {
    // Both tools evaluate the same artifact against the same source on the
    // same prompts; only the engine doing the forwards differs.
    let _runtime = journey_runtime();
    let reference_evaluation =
        read_json_document(&python_reference_directory(), "parity/kl_evaluation.json");
    let candidate_directory = python_reference_directory().join("optiq_mixed");
    let source_directory = source_model_directory();
    let tokenizer = HuggingFaceCalibrationTokenizer::from_model_directory(&candidate_directory)
        .expect("the artifact tokenizer should load");
    let prompt_count = json_count(&reference_evaluation["prompt_count"]);
    let prompts = tokenized_evaluation_prompts(
        &bundled_evaluation_prompts(prompt_count).expect("the bundled mix should parse"),
        &tokenizer,
        json_count(&reference_evaluation["sequence_length_tokens"]),
    )
    .expect("the prompts should tokenize");
    let view_parent = fresh_output_directory("parity-evaluation");
    let (reference_model_directory, _) = loadable_model_directory(&source_directory, &view_parent)
        .expect("the source should be loadable");
    let mut engine = Qwen3_5KlDivergenceEngine {
        memory_limits: journey_memory_limits(),
        prefill_chunk_tokens: PREFILL_CHUNK_TOKENS,
    };

    let summary = evaluate_kl_divergence(
        &mut engine,
        &KlEvaluationRequest {
            reference_directory: &reference_model_directory,
            candidate_directory: &candidate_directory,
            prompts: &prompts,
        },
        &mut StandardErrorProgress,
    )
    .expect("the evaluation should succeed");

    let figures = [
        ("mean", summary.mean_kl_divergence, "mean_kl_divergence"),
        (
            "median",
            summary.median_kl_divergence,
            "median_kl_divergence",
        ),
        (
            "95th percentile",
            summary.ninety_fifth_percentile_kl_divergence,
            "ninety_fifth_percentile_kl_divergence",
        ),
    ];
    for (figure_name, measured_figure, reference_key) in figures {
        let reference_figure = reference_evaluation[reference_key]
            .as_f64()
            .expect("a number");
        let relative_difference = (measured_figure - reference_figure).abs() / reference_figure;
        eprintln!(
            "[parity] KL evaluation {figure_name}: {measured_figure:.4} here, {reference_figure:.4} in Python ({:.1}% apart)",
            100.0 * relative_difference
        );
        assert!(relative_difference <= MAXIMUM_EVALUATION_RELATIVE_DIFFERENCE);
    }
    assert_eq!(summary.prompt_count, prompt_count);
    let _ = std::fs::remove_dir_all(&view_parent);
}

/// A non-negative whole number from a reference document.
fn json_count(count_value: &serde_json::Value) -> usize {
    usize::try_from(count_value.as_u64().expect("a count")).expect("the count fits")
}
