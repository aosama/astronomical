//! Unit tests for the sweep driver against a scripted probe model.
//!
//! The scripted probe records every call so the tests can assert on the
//! probe sequence (quantize, forward per sample, restore), on which widths
//! were skipped as free, on the measured-majority guard, and on checkpoint
//! resume, with no engine and no GPU.

use std::collections::BTreeMap;

use super::*;
use crate::bit_width::BitWidth;
use crate::module_inventory::{ModuleWeightSource, QuantizableModuleDescriptor, SourceTensorDtype};
use crate::module_taxonomy::ModuleRole;
use crate::progress::SilentProgressReporter;
use crate::sensitivity::probe::ModuleRestoreToken;

/// What the scripted probe was asked to do, in order.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeCall {
    Forward,
    SimulateQuantize { module_name: String, bits: u32 },
    Restore { module_name: String },
}

/// A probe whose "logits" are the (module, bits) currently simulated, so the
/// KL it reports is a deterministic function of the probe state.
struct ScriptedProbeModel {
    modules: Vec<QuantizableModuleDescriptor>,
    current_bit_widths: BTreeMap<String, BitWidth>,
    simulated: Option<(String, u32)>,
    /// Module names the probe pretends have no bf16 source.
    unresolvable_modules: Vec<String>,
    calls: Vec<ProbeCall>,
}

impl SensitivityProbeModel for ScriptedProbeModel {
    type Logits = Option<(String, u32)>;

    fn quantizable_modules(&self) -> Vec<QuantizableModuleDescriptor> {
        self.modules.clone()
    }

    fn current_bit_width(&self, module_name: &str) -> Option<BitWidth> {
        self.current_bit_widths.get(module_name).copied()
    }

    fn forward(
        &mut self,
        sample: &CalibrationForwardSample,
    ) -> Result<Self::Logits, QuantizationError> {
        let _ = sample;
        self.calls.push(ProbeCall::Forward);
        Ok(self.simulated.clone())
    }

    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError> {
        let _ = group_size;
        if self
            .unresolvable_modules
            .iter()
            .any(|name| name == module_name)
        {
            return Err(QuantizationError::ProbeModelFailure {
                description: format!("no bf16 source for {module_name}"),
            });
        }
        self.calls.push(ProbeCall::SimulateQuantize {
            module_name: module_name.to_owned(),
            bits: bit_width.bits(),
        });
        self.simulated = Some((module_name.to_owned(), bit_width.bits()));
        Ok(ModuleRestoreToken {
            module_name: module_name.to_owned(),
            engine_state: Box::new(()),
        })
    }

    fn restore_module(
        &mut self,
        restore_token: ModuleRestoreToken,
    ) -> Result<(), QuantizationError> {
        self.calls.push(ProbeCall::Restore {
            module_name: restore_token.module_name,
        });
        self.simulated = None;
        Ok(())
    }

    fn kl_divergence(
        &self,
        reference_logits: &Self::Logits,
        probe_logits: &Self::Logits,
    ) -> Result<f64, QuantizationError> {
        // Reference is always the unmodified state; a probe at b bits
        // reports 1 / b so lower widths look more sensitive.
        let _ = reference_logits;
        Ok(probe_logits
            .as_ref()
            .map_or(0.0, |(_, bits)| 1.0 / f64::from(*bits)))
    }
}

fn dense_module(module_name: &str) -> QuantizableModuleDescriptor {
    QuantizableModuleDescriptor {
        module_name: module_name.to_owned(),
        logical_shape: vec![64, 64],
        logical_parameter_count: 4096,
        source_dtype: SourceTensorDtype::BFloat16,
        weight_source: ModuleWeightSource::Whole {
            shard_file_name: "model.safetensors".to_owned(),
            tensor_name: format!("{module_name}.weight"),
        },
        role: ModuleRole::DenseFeedForward,
        transformer_block_index: crate::module_taxonomy::transformer_block_index(module_name),
    }
}

fn two_samples() -> Vec<CalibrationForwardSample> {
    vec![
        CalibrationForwardSample {
            token_ids: vec![1, 2, 3],
        },
        CalibrationForwardSample {
            token_ids: vec![4, 5, 6],
        },
    ]
}

#[test]
#[ignore = "pending implementation of run_exact_sweep"]
fn should_probe_every_module_at_every_width_and_restore_between_probes() {
    let mut probe_model = ScriptedProbeModel {
        modules: vec![dense_module("model.layers.0.mlp.up_proj")],
        current_bit_widths: BTreeMap::new(),
        simulated: None,
        unresolvable_modules: Vec::new(),
        calls: Vec::new(),
    };
    let samples = two_samples();
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("[4, 8] is a valid ladder");
    let mut silent_reporter = SilentProgressReporter;

    let table = run_exact_sweep(
        &mut probe_model,
        ExactSweepRequest {
            calibration_samples: &samples,
            candidate_bit_widths: &candidate_bit_widths,
            group_size: GroupSize::new(64).expect("64 is a valid group size"),
            method: SensitivityMethod::ExactBfloat16Reference,
            checkpoint_path: None,
            progress_reporter: &mut silent_reporter,
        },
    )
    .expect("the sweep should succeed");

    // Two reference forwards, then per width: one quantize, two forwards,
    // one restore.
    assert_eq!(
        probe_model.calls,
        vec![
            ProbeCall::Forward,
            ProbeCall::Forward,
            ProbeCall::SimulateQuantize {
                module_name: "model.layers.0.mlp.up_proj".to_owned(),
                bits: 4
            },
            ProbeCall::Forward,
            ProbeCall::Forward,
            ProbeCall::Restore {
                module_name: "model.layers.0.mlp.up_proj".to_owned()
            },
            ProbeCall::SimulateQuantize {
                module_name: "model.layers.0.mlp.up_proj".to_owned(),
                bits: 8
            },
            ProbeCall::Forward,
            ProbeCall::Forward,
            ProbeCall::Restore {
                module_name: "model.layers.0.mlp.up_proj".to_owned()
            },
        ]
    );
    let result = &table.results[0];
    assert_eq!(
        result.kl_divergence_at(BitWidth::new(4).expect("valid")),
        Some(0.25)
    );
    assert_eq!(
        result.kl_divergence_at(BitWidth::new(8).expect("valid")),
        Some(0.125)
    );
}

#[test]
#[ignore = "pending implementation of run_exact_sweep"]
fn should_record_zero_without_forwards_for_the_width_the_baseline_already_holds() {
    // Uniform-4-bit reference: probing 4 bits re-encodes the module to what
    // it already is, so the driver records 0 and spends no forwards.
    let mut probe_model = ScriptedProbeModel {
        modules: vec![dense_module("model.layers.0.mlp.up_proj")],
        current_bit_widths: BTreeMap::from([(
            "model.layers.0.mlp.up_proj".to_owned(),
            BitWidth::new(4).expect("valid"),
        )]),
        simulated: None,
        unresolvable_modules: Vec::new(),
        calls: Vec::new(),
    };
    let samples = two_samples();
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("[4, 8] is a valid ladder");
    let mut silent_reporter = SilentProgressReporter;

    let table = run_exact_sweep(
        &mut probe_model,
        ExactSweepRequest {
            calibration_samples: &samples,
            candidate_bit_widths: &candidate_bit_widths,
            group_size: GroupSize::new(64).expect("64 is a valid group size"),
            method: SensitivityMethod::ExactUniform4BitReference,
            checkpoint_path: None,
            progress_reporter: &mut silent_reporter,
        },
    )
    .expect("the sweep should succeed");

    let quantize_calls_at_four_bits = probe_model
        .calls
        .iter()
        .filter(|call| matches!(call, ProbeCall::SimulateQuantize { bits: 4, .. }))
        .count();
    assert_eq!(quantize_calls_at_four_bits, 0);
    assert_eq!(
        table.results[0].kl_divergence_at(BitWidth::new(4).expect("valid")),
        Some(0.0)
    );
}

#[test]
#[ignore = "pending implementation of run_exact_sweep"]
fn should_refuse_the_keep_bfloat16_tier_against_a_uniform_reference() {
    let mut probe_model = ScriptedProbeModel {
        modules: vec![dense_module("model.layers.0.mlp.up_proj")],
        current_bit_widths: BTreeMap::from([(
            "model.layers.0.mlp.up_proj".to_owned(),
            BitWidth::new(4).expect("valid"),
        )]),
        simulated: None,
        unresolvable_modules: Vec::new(),
        calls: Vec::new(),
    };
    let samples = two_samples();
    let candidate_bit_widths =
        CandidateBitWidths::new(&[4, 8, 16]).expect("[4, 8, 16] is a valid ladder");
    let mut silent_reporter = SilentProgressReporter;

    let sweep_outcome = run_exact_sweep(
        &mut probe_model,
        ExactSweepRequest {
            calibration_samples: &samples,
            candidate_bit_widths: &candidate_bit_widths,
            group_size: GroupSize::new(64).expect("64 is a valid group size"),
            method: SensitivityMethod::ExactUniform4BitReference,
            checkpoint_path: None,
            progress_reporter: &mut silent_reporter,
        },
    );

    assert!(matches!(
        sweep_outcome,
        Err(QuantizationError::KeepBfloat16TierNeedsBfloat16Reference)
    ));
}

#[test]
#[ignore = "pending implementation of require_measured_majority"]
fn should_refuse_a_sweep_that_skipped_most_modules_by_count() {
    // muse_glimmer scored 1 of 417 modules and still shipped a quant.
    let guard_outcome = require_measured_majority(416, 417, 0, 1);

    assert!(matches!(
        guard_outcome,
        Err(QuantizationError::SensitivitySweepMeasuredTooLittle { .. })
    ));
}

#[test]
#[ignore = "pending implementation of require_measured_majority"]
fn should_refuse_a_sweep_that_skipped_most_parameters_by_mass() {
    // Nemotron-H skipped 46 of 164 modules (28%, under the count rule) but
    // they held most of the parameters.
    let guard_outcome = require_measured_majority(46, 164, 24_000_000_000, 30_000_000_000);

    assert!(matches!(
        guard_outcome,
        Err(QuantizationError::SensitivitySweepMeasuredTooLittle { .. })
    ));
}

#[test]
#[ignore = "pending implementation of require_measured_majority"]
fn should_accept_a_sweep_with_a_few_unmatched_modules() {
    assert!(require_measured_majority(3, 417, 1_000_000, 9_000_000_000).is_ok());
}

#[test]
#[ignore = "pending implementation of mean_kl_divergence"]
fn should_average_per_sample_divergences() {
    assert_eq!(mean_kl_divergence(&[0.2, 0.4]), 0.3);
    assert_eq!(mean_kl_divergence(&[]), 0.0);
}
