//! The Qwen3.5 probe: one resident model whose modules can be swapped.

use std::collections::BTreeMap;
use std::path::Path;

use astronomical_mlx_c_rust::MlxArray;
use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::module_inventory::QuantizableModuleDescriptor;
use astronomical_quantization::sensitivity::{
    CalibrationForwardSample, ModuleRestoreToken, SensitivityProbeModel,
};

use crate::qwen3_5_resident::Qwen3_5ResidentModel;

/// Where a module's weight lives inside the typed weights, so a swap can
/// reach it without string matching on every probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModuleLocation {
    Embedding,
    LanguageModelHead,
    /// `decoder_layer_index`, then which affine module inside the layer.
    DecoderLayerAttention {
        decoder_layer_index: usize,
        projection: AttentionProjection,
    },
    DecoderLayerDenseFeedForward {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
    /// Fused resident expert arrays for one layer and projection.
    DecoderLayerRoutedExperts {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
    DecoderLayerSharedExpert {
        decoder_layer_index: usize,
        projection: FeedForwardProjection,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionProjection {
    Query,
    Key,
    Value,
    Output,
    LinearInputQueriesKeysValues,
    LinearOutputGate,
    LinearUpdateRate,
    LinearDecayInterval,
    LinearOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardProjection {
    Gate,
    Up,
    Down,
}

/// The probe over one loaded resident model.
pub struct Qwen3_5SensitivityProbe {
    model: Qwen3_5ResidentModel,
    /// Canonical artifact module name → where it lives, built once at load
    /// from the validated artifact's tensor inventory so names match what
    /// the quantization crate's inventory produces for the same directory.
    module_locations: BTreeMap<String, ModuleLocation>,
    /// Present on the uniform-baseline path: the bf16 source directory whose
    /// shards are memory-mapped and read per module.
    bfloat16_source_directory: Option<std::path::PathBuf>,
    /// The prompt chunk the forward uses: fixed at 2,048 tokens, the
    /// repository's reproducibility rule.
    prefill_chunk_tokens: usize,
}

/// Loads a resident model from `model_directory` for probing.
///
/// Work to implement: validate the artifact with the same validator the
/// daemon uses, initialize an MLX runtime sized to the artifact plus the
/// encoder's working copies (the caller supplies the memory ceiling),
/// `Qwen3_5ResidentModel::load` then
/// `materialize_complete_weights_matching_production`, and build the module
/// location map by walking `Qwen3_5Weights` (embedding, per-layer attention
/// and feed-forward affine modules, resident expert arrays, head). When
/// `bfloat16_source_directory` is given the loaded model is a uniform
/// quantized baseline and every probe re-encodes a streamed bf16 weight;
/// when it is absent the loaded model is itself bf16 and every probe
/// quantize-dequantizes in place.
pub fn load_qwen3_5_probe_model(
    model_directory: &Path,
    bfloat16_source_directory: Option<&Path>,
    prefill_chunk_tokens: usize,
) -> Result<Qwen3_5SensitivityProbe, QuantizationError> {
    let _ = (
        model_directory,
        bfloat16_source_directory,
        prefill_chunk_tokens,
    );
    todo!("validate, initialize the runtime, load and materialize, index module locations")
}

impl Qwen3_5SensitivityProbe {
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.module_locations.len()
    }

    #[must_use]
    pub const fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    #[must_use]
    pub fn bfloat16_source_directory(&self) -> Option<&Path> {
        self.bfloat16_source_directory.as_deref()
    }

    pub(crate) fn model(&self) -> &Qwen3_5ResidentModel {
        &self.model
    }
}

/// The engine-private restore state: the arrays that were swapped out.
struct SwappedWeights {
    previous: crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights,
}

impl SensitivityProbeModel for Qwen3_5SensitivityProbe {
    /// Final-position logits for every position of the sample, on device.
    type Logits = MlxArray;

    /// Work to implement: one descriptor per module location, with the
    /// logical shape and parameter count taken from the resident arrays
    /// (for a quantized module, rebuilt from the scales' last dimension
    /// times the group size, never from the packed array), the role from
    /// the quantization crate's taxonomy, and the block index from the
    /// decoder layer index.
    fn quantizable_modules(&self) -> Vec<QuantizableModuleDescriptor> {
        let _ = self.model();
        todo!("walk module_locations and describe each module")
    }

    /// `Some(bits)` for a `Quantized` affine module, `None` for bf16.
    fn current_bit_width(&self, module_name: &str) -> Option<BitWidth> {
        let _ = module_name;
        todo!("inspect the affine weights at the module's location")
    }

    /// Work to implement: a fresh request decoder state matching the loaded
    /// weights; `prefill_chunk` over every full 2,048-token chunk but the
    /// last, then `forward_chunk` on the final chunk, returning its logits.
    /// Because KL needs logits at every position, the last chunk must be
    /// forwarded through the logits-returning path for the whole chunk, not
    /// only the final token; confirm `forward_chunk` returns `[tokens,
    /// vocabulary]` and widen it if it returns only the last row.
    fn forward(
        &mut self,
        sample: &CalibrationForwardSample,
    ) -> Result<Self::Logits, QuantizationError> {
        let _ = sample;
        todo!("prefill then forward the final chunk and return its logits")
    }

    /// Work to implement, bf16 path: take the module's `NativeBfloat16`
    /// weight, `quantize_affine` then `dequantize_affine` at `(bit_width,
    /// group_size)`, cast back to bfloat16, and swap it in, keeping the
    /// previous array in the token. Uniform-baseline path: memory-map the
    /// bf16 shard named by the module's source (through the runtime's
    /// bounded safetensors loader), assemble halves or stacks as the
    /// inventory's `ModuleWeightSource` says, `quantize_affine` at the
    /// requested width, and swap in packed weight, scales, biases, bits,
    /// and group size. The keep-bfloat16 tier is rejected on the baseline
    /// path. Evaluate the new arrays before returning.
    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError> {
        let _ = (module_name, bit_width, group_size);
        todo!("swap the module's affine weights for a simulate-quantized copy")
    }

    /// Puts the previous arrays back and evaluates them.
    fn restore_module(
        &mut self,
        restore_token: ModuleRestoreToken,
    ) -> Result<(), QuantizationError> {
        let swapped = restore_token
            .engine_state
            .downcast::<SwappedWeights>()
            .map_err(|_| QuantizationError::ProbeModelFailure {
                description: format!(
                    "restore token for {} did not carry Qwen3.5 swapped weights",
                    restore_token.module_name
                ),
            })?;
        let _ = swapped.previous;
        todo!("write the previous affine weights back at the module's location")
    }

    /// Work to implement: cast both to float32, `log_softmax` along the last
    /// axis, `softmax` of the reference, reduce
    /// `sum(p_ref * (log_ref - log_probe))` over the vocabulary and `mean`
    /// over positions, evaluate, and read the scalar back as f64. Reference
    /// logits may be stored as float16 to halve memory; the math stays
    /// float32.
    fn kl_divergence(
        &self,
        reference_logits: &Self::Logits,
        probe_logits: &Self::Logits,
    ) -> Result<f64, QuantizationError> {
        let _ = (reference_logits, probe_logits);
        todo!("float32 log-softmax KL on device, scalar read back")
    }

    /// Drains the allocator cache: large fused expert tensors otherwise
    /// accumulate in MLX's page cache and the process is killed within
    /// about ten modules.
    fn release_transient_memory(&mut self) -> Result<(), QuantizationError> {
        self.model
            .runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()
            .map_err(|runtime_error| QuantizationError::MlxRuntimeFailure {
                operation: "drain the allocator cache between probes",
                description: runtime_error.to_string(),
            })
    }
}
