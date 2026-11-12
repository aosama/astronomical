//! The exact KL sweep through the model-serving probe on a real base.

use std::collections::BTreeMap;
use std::path::Path;

use astronomical_model_serving::quantization_probe::{
    Qwen3_5SensitivityProbe, load_qwen3_5_probe_model,
};
use astronomical_quantization::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::module_inventory::QuantizableModuleDescriptor;
use astronomical_quantization::pipeline::probe_source_view::loadable_model_directory;
use astronomical_quantization::pipeline::{
    PipelineMethod, ProbeModelHandle, ProbeModelLoader, ProbeModelSweeper,
    QuantizationPipelineRequest, ReferenceMode, run_quantization_pipeline,
};
use astronomical_quantization::sensitivity::exact_sweep::{ExactSweepRequest, run_exact_sweep};
use astronomical_quantization::sensitivity::{
    CalibrationForwardSample, ModuleRestoreToken, SensitivityMethod, SensitivityProbeModel,
    SensitivityTable,
};
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::served_generation::{generate_greedy_continuation, romeo_and_juliet_prompt};
use super::support::{
    PREFILL_CHUNK_TOKENS, StandardErrorProgress, bfloat16_copy_of, fresh_output_directory,
    journey_memory_limits, journey_runtime, source_model_directory,
};

/// Short sequences keep a full sweep of a small base inside the journey
/// budget; the signal is per position, so length trades precision for time.
const CALIBRATION_SEQUENCE_LENGTH_TOKENS: usize = 256;
const CALIBRATION_SEQUENCE_COUNT: usize = 2;
const MODULES_BEFORE_INTERRUPTION: usize = 3;
const CHECKPOINT_FILE_NAME: &str = "sensitivity_checkpoint.json";

/// 3 and 8 bits rather than OptiQ's default 4 and 8: on a small bf16 base
/// every probe at 4 bits and above lands on the network's own bf16
/// activation-noise floor (about 1e-3 KL on Qwen3.5 0.8B, not monotone in
/// width), so only a ladder whose low rung clears the floor makes
/// monotonicity a meaningful assertion.
const JOURNEY_CANDIDATE_BITS: [u32; 2] = [3, 8];

#[test]
#[serial]
#[ignore = "needs model_directory (a small Qwen3.5 checkpoint) in the explicit test-run configuration"]
fn should_measure_sensitivity_with_the_bfloat16_reference_and_resume_from_a_checkpoint() {
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("exact");
    let bfloat16_source = bfloat16_copy_of(
        &runtime,
        &source_model_directory(),
        &output_directory.join("bfloat16_source"),
    );
    let calibration_samples = romeo_and_juliet_samples(&bfloat16_source);
    let checkpoint_path = output_directory.join(CHECKPOINT_FILE_NAME);
    let probe_model = load_probe(&bfloat16_source);
    let mut interrupting_probe =
        CountingProbe::interrupting_after(probe_model, MODULES_BEFORE_INTERRUPTION);

    let interrupted = sweep(
        &mut interrupting_probe,
        &calibration_samples,
        &checkpoint_path,
    );
    let first_modules = interrupting_probe.restored_module_names.clone();
    let mut resumed_probe = CountingProbe::uninterrupted(interrupting_probe.inner);
    let sensitivity_table = sweep(&mut resumed_probe, &calibration_samples, &checkpoint_path)
        .expect("the resumed sweep should pass the measured-majority guard");

    assert!(interrupted.is_err());
    assert_eq!(first_modules.len(), MODULES_BEFORE_INTERRUPTION);
    for module_name in &first_modules {
        assert_eq!(
            resumed_probe.simulate_count(module_name),
            0,
            "{module_name} was re-probed"
        );
    }
    assert_monotone_in_width(&sensitivity_table);
    drop(resumed_probe);

    let artifact_directory = convert_resuming_the_checkpoint(&bfloat16_source, &output_directory);
    let continuation = generate_greedy_continuation(&artifact_directory);
    eprintln!("[exact-sweep] continuation={continuation:?}");
    assert!(!continuation.trim().is_empty());
    let _ = std::fs::remove_dir_all(&output_directory);
}

/// A single-file bf16 source has no shard index for the engine to read, so
/// the probe loads through the same indexed view the pipeline builds. The
/// view is hard links in a temporary directory and costs no disk.
fn load_probe(model_directory: &Path) -> Qwen3_5SensitivityProbe {
    let view_parent = fresh_output_directory("probe-view");
    let (probe_directory, _) = loadable_model_directory(model_directory, &view_parent)
        .expect("the probe directory should be available");
    load_qwen3_5_probe_model(
        &probe_directory,
        None,
        journey_memory_limits(),
        PREFILL_CHUNK_TOKENS,
    )
    .expect("the bf16 probe model should load")
}

fn sweep(
    probe_model: &mut CountingProbe,
    calibration_samples: &[CalibrationForwardSample],
    checkpoint_path: &Path,
) -> Result<SensitivityTable, QuantizationError> {
    let candidate_bit_widths =
        CandidateBitWidths::new(&JOURNEY_CANDIDATE_BITS).expect("valid ladder");
    run_exact_sweep(
        probe_model,
        ExactSweepRequest {
            calibration_samples,
            candidate_bit_widths: &candidate_bit_widths,
            group_size: GroupSize::new(64).expect("valid"),
            method: SensitivityMethod::ExactBfloat16Reference,
            checkpoint_path: Some(checkpoint_path),
            progress_reporter: &mut StandardErrorProgress,
        },
    )
}

/// Every KL finite, and 8 bits no worse than 3 for at least 95% of
/// modules: across rungs that clear the noise floor the signal is monotone
/// in width, so a reversal means the swap is not reaching the forward.
fn assert_monotone_in_width(sensitivity_table: &SensitivityTable) {
    let (low_bits, eight_bits) = (
        BitWidth::new(3).expect("valid"),
        BitWidth::new(8).expect("valid"),
    );
    let mut monotone_module_count = 0;
    for result in &sensitivity_table.results {
        let kl_at_low = result.kl_divergence_by_bit_width[&low_bits];
        let kl_at_eight = result.kl_divergence_by_bit_width[&eight_bits];
        assert!(
            kl_at_low.is_finite() && kl_at_eight.is_finite(),
            "{}",
            result.module_name
        );
        if kl_at_eight <= kl_at_low {
            monotone_module_count += 1;
        }
    }
    let module_count = sensitivity_table.results.len();
    eprintln!("[exact-sweep] monotone={monotone_module_count}/{module_count}");
    assert!(monotone_module_count as f64 >= 0.95 * module_count as f64);
}

/// Runs the pipeline at 4.5 bits per weight against the same checkpoint,
/// so every module resumes and only the reference forwards run.
fn convert_resuming_the_checkpoint(
    bfloat16_source: &Path,
    output_directory: &Path,
) -> std::path::PathBuf {
    let request = QuantizationPipelineRequest {
        source_directory: bfloat16_source.to_path_buf(),
        output_directory: output_directory.to_path_buf(),
        base_model_name: "acceptance/exact".to_owned(),
        method: PipelineMethod::ExactKlSensitivity,
        reference_mode: ReferenceMode::Bfloat16,
        target_bits_per_weight: 4.5,
        candidate_bits: JOURNEY_CANDIDATE_BITS.to_vec(),
        group_size: GroupSize::new(64).expect("valid"),
        calibration_sequence_count: CALIBRATION_SEQUENCE_COUNT,
        calibration_sequence_length_tokens: CALIBRATION_SEQUENCE_LENGTH_TOKENS,
        calibration_mix_path: None,
        floor_components_per_block: 2,
        range_search_policy_enabled: true,
        range_search_maximum_bits: 3,
        overwrite_output: false,
    };
    let runtime = journey_runtime();
    let outcome = run_quantization_pipeline(
        &runtime,
        &request,
        u64::MAX,
        &mut JourneyProbeLoader,
        &mut StandardErrorProgress,
    )
    .expect("the exact conversion should succeed");
    assert_eq!(outcome.method, SensitivityMethod::ExactBfloat16Reference);
    assert!(outcome.sensitivity_document_written);
    outcome.artifact_directory
}

struct JourneyProbeLoader;

impl ProbeModelLoader for JourneyProbeLoader {
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

/// Fixed-length Romeo and Juliet sequences through the source's tokenizer.
pub(super) fn romeo_and_juliet_samples(model_directory: &Path) -> Vec<CalibrationForwardSample> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the tokenizer should load");
    let mut prompt_text = String::new();
    for _ in 0..4 {
        prompt_text.push_str(&romeo_and_juliet_prompt());
    }
    let token_ids = tokenizer
        .encode(prompt_text, false)
        .expect("the fixture should encode")
        .get_ids()
        .to_vec();
    let mut calibration_samples = Vec::with_capacity(CALIBRATION_SEQUENCE_COUNT);
    for sequence_tokens in token_ids
        .chunks_exact(CALIBRATION_SEQUENCE_LENGTH_TOKENS)
        .take(CALIBRATION_SEQUENCE_COUNT)
    {
        calibration_samples.push(CalibrationForwardSample {
            token_ids: sequence_tokens.to_vec(),
        });
    }
    assert_eq!(calibration_samples.len(), CALIBRATION_SEQUENCE_COUNT);
    calibration_samples
}

/// Delegates to the real probe, counting simulate calls per module and
/// optionally failing the forward once enough modules finished, which
/// aborts the sweep the way a killed process would.
struct CountingProbe {
    inner: Qwen3_5SensitivityProbe,
    interrupt_after_module_count: Option<usize>,
    simulate_count_by_module: BTreeMap<String, usize>,
    restored_module_names: Vec<String>,
}

impl CountingProbe {
    fn interrupting_after(inner: Qwen3_5SensitivityProbe, module_count: usize) -> Self {
        Self {
            inner,
            interrupt_after_module_count: Some(module_count),
            simulate_count_by_module: BTreeMap::new(),
            restored_module_names: Vec::new(),
        }
    }

    fn uninterrupted(inner: Qwen3_5SensitivityProbe) -> Self {
        Self {
            interrupt_after_module_count: None,
            ..Self::interrupting_after(inner, 0)
        }
    }

    fn simulate_count(&self, module_name: &str) -> usize {
        self.simulate_count_by_module
            .get(module_name)
            .copied()
            .unwrap_or(0)
    }
}

impl SensitivityProbeModel for CountingProbe {
    type Logits = MlxArray;

    fn quantizable_modules(&self) -> Vec<QuantizableModuleDescriptor> {
        self.inner.quantizable_modules()
    }

    fn current_bit_width(&self, module_name: &str) -> Option<BitWidth> {
        self.inner.current_bit_width(module_name)
    }

    fn forward(
        &mut self,
        sample: &CalibrationForwardSample,
    ) -> Result<MlxArray, QuantizationError> {
        self.inner.forward(sample)
    }

    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError> {
        let is_new_module = !self.simulate_count_by_module.contains_key(module_name);
        let should_interrupt = is_new_module
            && self.interrupt_after_module_count == Some(self.simulate_count_by_module.len());
        if should_interrupt {
            // Not a ProbeModelFailure: the driver skips a module on that
            // and keeps going, while any other error aborts the sweep the
            // way a killed process would. Raised before the swap, so the
            // interrupted module was never altered.
            return Err(QuantizationError::MlxRuntimeFailure {
                operation: "the simulated interruption",
                description: "the sweep was stopped after three modules".to_owned(),
            });
        }
        *self
            .simulate_count_by_module
            .entry(module_name.to_owned())
            .or_insert(0) += 1;
        self.inner
            .simulate_quantize_module(module_name, bit_width, group_size)
    }

    fn restore_module(
        &mut self,
        restore_token: ModuleRestoreToken,
    ) -> Result<(), QuantizationError> {
        if self.restored_module_names.last() != Some(&restore_token.module_name) {
            self.restored_module_names
                .push(restore_token.module_name.clone());
        }
        self.inner.restore_module(restore_token)
    }

    fn kl_divergence(
        &self,
        reference_logits: &MlxArray,
        probe_logits: &MlxArray,
    ) -> Result<f64, QuantizationError> {
        self.inner.kl_divergence(reference_logits, probe_logits)
    }

    fn release_transient_memory(&mut self) -> Result<(), QuantizationError> {
        self.inner.release_transient_memory()
    }
}

#[test]
#[serial]
#[ignore = "needs model_directory (a small Qwen3.5 checkpoint) in the explicit test-run configuration"]
fn should_reproduce_its_own_reference_forward() {
    // Every probe's KL is measured against reference logits from an earlier
    // forward, so a forward that drifts between calls puts a floor under
    // every measurement. Identical weights must give identical logits.
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("reproducible");
    let bfloat16_source = bfloat16_copy_of(
        &runtime,
        &source_model_directory(),
        &output_directory.join("bfloat16_source"),
    );
    let calibration_samples = romeo_and_juliet_samples(&bfloat16_source);
    let mut probe_model = load_probe(&bfloat16_source);

    let first_logits = probe_model
        .forward(&calibration_samples[0])
        .expect("forward");
    let second_logits = probe_model
        .forward(&calibration_samples[0])
        .expect("forward");
    let third_logits = probe_model
        .forward(&calibration_samples[0])
        .expect("forward");

    let first_to_second = probe_model
        .kl_divergence(&first_logits, &second_logits)
        .expect("kl");
    let second_to_third = probe_model
        .kl_divergence(&second_logits, &third_logits)
        .expect("kl");
    eprintln!(
        "[reproducible-forward] first_to_second={first_to_second:.3e} second_to_third={second_to_third:.3e}"
    );
    assert!(first_to_second < 1e-6);
    assert!(second_to_third < 1e-6);

    // A 2-bit probe must move the output far more than an 8-bit one (the
    // swap reaches the forward), and its restore must leave the model
    // exactly as it was.
    let module_name = "language_model.model.layers.4.mlp.up_proj";
    let eight_bit_kl = probed_kl(
        &mut probe_model,
        module_name,
        8,
        &calibration_samples[0],
        &first_logits,
    );
    let two_bit_kl = probed_kl(
        &mut probe_model,
        module_name,
        2,
        &calibration_samples[0],
        &first_logits,
    );
    eprintln!("[reproducible-forward] {module_name} kl@8={eight_bit_kl:.3e} kl@2={two_bit_kl:.3e}");
    assert!(two_bit_kl > 10.0 * eight_bit_kl);
    let restored_logits = probe_model
        .forward(&calibration_samples[0])
        .expect("forward");
    let after_restore = probe_model
        .kl_divergence(&first_logits, &restored_logits)
        .expect("kl");
    eprintln!("[reproducible-forward] after_restore={after_restore:.3e}");
    assert!(after_restore < 1e-6);
    let _ = std::fs::remove_dir_all(&output_directory);
}

/// Modules at the first, a middle and the last decoder layer of the small
/// base: the resume points that reuse the least, about half, and nearly all
/// of the reference forward.
const RESUMED_FORWARD_MODULE_NAMES: [&str; 3] = [
    "language_model.model.layers.0.mlp.up_proj",
    "language_model.model.layers.11.mlp.down_proj",
    "language_model.model.layers.23.mlp.up_proj",
];

#[test]
#[serial]
#[ignore = "needs model_directory (a small Qwen3.5 checkpoint) in the explicit test-run configuration"]
fn should_resume_a_probe_forward_to_the_same_logits_as_the_whole_model_forward() {
    // The sweep's forward resumes from the reference forward's recorded
    // layer inputs. Every sensitivity value rests on that being the whole
    // forward's arithmetic, so a resumed probe must equal a probe that
    // recomputed every layer, while still differing from the reference.
    let runtime = journey_runtime();
    let output_directory = fresh_output_directory("resumed-forward");
    let bfloat16_source = bfloat16_copy_of(
        &runtime,
        &source_model_directory(),
        &output_directory.join("bfloat16_source"),
    );
    let calibration_samples = romeo_and_juliet_samples(&bfloat16_source);
    let calibration_sample = &calibration_samples[0];
    let mut probe_model = load_probe(&bfloat16_source);
    let reference_logits = probe_model.forward(calibration_sample).expect("forward");

    for module_name in RESUMED_FORWARD_MODULE_NAMES {
        let restore_token = probe_model
            .simulate_quantize_module(
                module_name,
                BitWidth::new(2).expect("valid"),
                GroupSize::new(64).expect("valid"),
            )
            .expect("simulate");
        let resumed_logits = probe_model.forward(calibration_sample).expect("forward");
        let whole_model_logits = probe_model
            .whole_model_logits(&calibration_sample.token_ids)
            .expect("whole-model forward");
        let resumed_to_whole = probe_model
            .kl_divergence(&whole_model_logits, &resumed_logits)
            .expect("kl");
        let reference_to_resumed = probe_model
            .kl_divergence(&reference_logits, &resumed_logits)
            .expect("kl");
        probe_model.restore_module(restore_token).expect("restore");
        eprintln!(
            "[resumed-forward] {module_name} resumed_to_whole={resumed_to_whole:.3e} reference_to_resumed={reference_to_resumed:.3e}"
        );
        assert!(resumed_to_whole < 1e-9, "{module_name} resumed differently");
        assert!(
            reference_to_resumed > 1e-6,
            "{module_name} swap did not reach the forward"
        );
    }
    let _ = std::fs::remove_dir_all(&output_directory);
}

/// KL against `reference_logits` with one module probed at `bits`, restored
/// afterwards.
pub(super) fn probed_kl(
    probe_model: &mut Qwen3_5SensitivityProbe,
    module_name: &str,
    bits: u32,
    calibration_sample: &CalibrationForwardSample,
    reference_logits: &MlxArray,
) -> f64 {
    let restore_token = probe_model
        .simulate_quantize_module(
            module_name,
            BitWidth::new(bits).expect("valid"),
            GroupSize::new(64).expect("valid"),
        )
        .expect("simulate");
    let probed_logits = probe_model.forward(calibration_sample).expect("forward");
    let probed_kl = probe_model
        .kl_divergence(reference_logits, &probed_logits)
        .expect("kl");
    probe_model.restore_module(restore_token).expect("restore");
    probed_kl
}
