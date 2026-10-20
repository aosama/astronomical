//! The seam between the sweep driver and a model engine.
//!
//! OptiQ mutates mlx-lm module attributes in place (`module.weight = ...`)
//! and reads `model(*args).logits`. Astronomical's engines hold typed weights
//! and do not expose Python-style attribute swapping, so the port inverts the
//! dependency: this crate defines what a probe model must be able to do, and
//! each engine implements it where it has access to its own weights.
//! model-serving provides the Qwen3.5 implementation behind its `direct-mlx`
//! feature; the offline tests here use a scripted fake.
//!
//! Logits never cross this boundary as a concrete array type, which is what
//! keeps the sweep driver free of MLX. The probe returns an opaque
//! `Self::Logits` and also computes the KL divergence between two of them, so
//! the GPU implementation can keep both tensors on device and reduce there.

use crate::bit_width::{BitWidth, GroupSize};
use crate::error::QuantizationError;
use crate::module_inventory::QuantizableModuleDescriptor;

/// One calibration sequence as the probe wants it: token ids of a fixed
/// length, batch size one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationForwardSample {
    pub token_ids: Vec<u32>,
}

/// Proof that a module was altered, handed back to restore it. Making the
/// restore explicit (rather than relying on the driver to remember) means a
/// probe error mid-module still leaves the model restorable, which OptiQ's
/// `try`-less in-place swaps do not guarantee.
#[derive(Debug)]
pub struct ModuleRestoreToken {
    pub module_name: String,
    /// Engine-private restore state, boxed so the trait stays object-safe.
    pub engine_state: Box<dyn std::any::Any + Send>,
}

/// A loaded model that can be probed one module at a time.
///
/// Implementation notes for model-serving's Qwen3.5 adapter:
///
/// - `quantizable_modules` walks `Qwen3_5Weights` (embedding, each decoder
///   layer's attention and feed-forward `Qwen3_5AffineWeights`, the head) and
///   maps each to the canonical name the artifact uses; fused resident expert
///   arrays map to the `switch_mlp.{gate,up,down}_proj` names with the
///   `[n_experts, out, in]` logical shape.
/// - `simulate_quantize_module` on a `NativeBfloat16` weight runs
///   `quantize_affine` then `dequantize_affine` and swaps the dequantized
///   bf16 array in (the bf16-reference path). On a `Quantized` weight whose
///   bf16 source is streamed from a sibling directory it re-encodes the
///   streamed bf16 weight at the requested width and swaps all three arrays
///   plus bits and group size (the uniform-baseline path). Either way the
///   previous arrays go into the restore token.
/// - Forward passes use `forward_chunk` on a fresh request decoder state per
///   sample with the prompt chunk size fixed at 2,048 tokens, the repository's
///   reproducibility rule, and return the final-position logits for every
///   position of the sample.
/// - `kl_divergence` casts both logit tensors to float32, log-softmaxes along
///   the vocabulary axis, and averages `sum(p_ref * (log p_ref - log p_probe))`
///   over positions, exactly OptiQ's `_kl_from_ref`. Reference logits may be
///   stored as float16 to halve memory; the math is float32 regardless.
/// - After every module the adapter must drain the allocator cache
///   (`synchronize_gpu_stream_and_clear_allocator_cache`): large fused expert
///   tensors otherwise accumulate in MLX's page cache and macOS kills the
///   process within about ten modules.
pub trait SensitivityProbeModel {
    /// Opaque logits for one forward pass; stays on device for GPU engines.
    type Logits;

    /// Every module the sweep may probe, in a stable order.
    fn quantizable_modules(&self) -> Vec<QuantizableModuleDescriptor>;

    /// The bit width the running model already holds for `module_name`, or
    /// `None` for a bf16 module. The driver skips probing a width equal to
    /// the baseline's own width: re-encoding a 4-bit module at 4 bits gives
    /// KL of zero by construction, and skipping it halves a `[4, 8]` sweep.
    fn current_bit_width(&self, module_name: &str) -> Option<BitWidth>;

    /// Forwards one sample through the model in its current state.
    fn forward(
        &mut self,
        sample: &CalibrationForwardSample,
    ) -> Result<Self::Logits, QuantizationError>;

    /// Replaces `module_name`'s weight with a version quantized at `bit_width`
    /// (then dequantized back to the source dtype for bf16 references), and
    /// returns the token that restores the original.
    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError>;

    /// Puts the original weight back.
    fn restore_module(
        &mut self,
        restore_token: ModuleRestoreToken,
    ) -> Result<(), QuantizationError>;

    /// KL(reference ‖ probe), averaged over batch and positions, as float64.
    fn kl_divergence(
        &self,
        reference_logits: &Self::Logits,
        probe_logits: &Self::Logits,
    ) -> Result<f64, QuantizationError>;

    /// Releases transient device memory between modules. The default is a
    /// no-op for CPU fakes.
    fn release_transient_memory(&mut self) -> Result<(), QuantizationError> {
        Ok(())
    }
}
