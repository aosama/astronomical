//! The Qwen3.5 probe: one resident model whose modules can be swapped.
//!
//! The swap-forward-restore cycle is ported from OptiQ's `_exact_with_bf16_reference` (`optiq/core/sensitivity.py`); the model side is Astronomical's own Qwen3.5 engine.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::module_inventory::{ModuleInventory, QuantizableModuleDescriptor};
use astronomical_quantization::requantize::source_tensor_reader::{
    SourceShardMaps, read_module_weight,
};
use astronomical_quantization::sensitivity::{
    CalibrationForwardSample, ModuleRestoreToken, SensitivityProbeModel,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_resident::Qwen3_5ResidentModel;
use crate::{Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration};

use super::expert_swaps::SwappedOutWeights;
use super::module_locations::{FeedForwardProjection, ModuleLocation, module_location_for};
use super::simulated_quantization::{
    encoded_from_source, mean_kl_divergence, mlx_step, per_position_kl_divergences,
    quantize_dequantized_bfloat16,
};

#[path = "module_swaps.rs"]
mod module_swaps;

/// The production standard chunking tuple the daemon resolves for the
/// resident engine, so probes run the shapes a served request runs.
const FULL_ATTENTION_KV_GROWTH_TOKENS: u32 = 256;
const PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
const SSD_PAGING_PREFILL_LAYER_INTERVAL: u32 = 1;
const SSD_PAGING_GENERATION_LAYER_INTERVAL: u32 = 3;

/// The validator's output budget; a probe never decodes, so any positive
/// figure the artifact accepts is equivalent.
const PROBE_MAXIMUM_OUTPUT_TOKENS: u32 = 4_096;

/// OptiQ's default group size, used to inventory the probed directory the
/// same way the pipeline inventories it.
const PROBE_INVENTORY_GROUP_SIZE: u32 = 64;

/// The probe over one loaded resident model.
pub struct Qwen3_5SensitivityProbe {
    model: Qwen3_5ResidentModel,
    /// Every quantizable module of the probed directory, as the quantization
    /// crate's inventory names it, so the sweep's names match the pipeline's.
    module_descriptors: Vec<QuantizableModuleDescriptor>,
    /// Canonical module name → where its affine weights live.
    module_locations: BTreeMap<String, ModuleLocation>,
    /// Present on the uniform-baseline path: where each module's bf16
    /// source tensor lives, and the shards opened so far.
    bfloat16_source: Option<Bfloat16Source>,
    /// The prompt chunk the forward uses: 2,048 tokens by the repository's
    /// reproducibility rule.
    prefill_chunk_tokens: usize,
    /// The arrays a probe swapped out, keyed by module name, until restore.
    /// They live here rather than in the restore token because the token's
    /// state must be `Send` and MLX arrays are not.
    swapped_out_weights: BTreeMap<String, SwappedOutWeights>,
}

/// The bf16 source a uniform-baseline probe re-encodes modules from, read
/// through the requantizer's own reader so a probe sees exactly the weight
/// the final conversion will encode: whole tensors, the halves of a fused
/// gate-up expert tensor, and per-expert tensors stacked in expert order.
struct Bfloat16Source {
    directory: PathBuf,
    inventory: ModuleInventory,
    shard_maps: SourceShardMaps,
}

/// Loads a resident model from `model_directory` for probing: validated by
/// the daemon's own validator, loaded and materialized as production does,
/// under `memory_limits` (the runtime is process-global, so these must match
/// any runtime the caller already initialized). With
/// `bfloat16_source_directory` the loaded model is a uniform quantized
/// baseline and every probe re-encodes the module's streamed bf16 source;
/// without it the loaded model is itself bf16 and every probe
/// quantize-dequantizes in place.
pub fn load_qwen3_5_probe_model(
    model_directory: &Path,
    bfloat16_source_directory: Option<&Path>,
    memory_limits: MlxMemoryLimits,
    prefill_chunk_tokens: usize,
) -> Result<Qwen3_5SensitivityProbe, QuantizationError> {
    let inventory_group_size = GroupSize::new(PROBE_INVENTORY_GROUP_SIZE)?;
    let inventory = ModuleInventory::from_source_directory(model_directory, inventory_group_size)?;
    let bfloat16_source = match bfloat16_source_directory {
        Some(source_directory) => Some(Bfloat16Source {
            directory: source_directory.to_path_buf(),
            inventory: ModuleInventory::from_source_directory(
                source_directory,
                inventory_group_size,
            )?,
            shard_maps: SourceShardMaps::new(source_directory),
        }),
        None => None,
    };
    let model = load_resident_model(model_directory, memory_limits)?;
    let mut module_descriptors = Vec::new();
    let mut module_locations = BTreeMap::new();
    // The allocation set, not every module: OptiQ's sweep walks linear
    // layers only, so the embedding table is never probed and takes the
    // allocation's highest width when the artifact is written.
    for descriptor in inventory.allocation_modules() {
        if let Some(location) = module_location_for(&descriptor.module_name) {
            module_locations.insert(descriptor.module_name.clone(), location);
        }
        module_descriptors.push(descriptor.clone());
    }
    Ok(Qwen3_5SensitivityProbe {
        model,
        module_descriptors,
        module_locations,
        bfloat16_source,
        prefill_chunk_tokens,
        swapped_out_weights: BTreeMap::new(),
    })
}

fn load_resident_model(
    model_directory: &Path,
    memory_limits: MlxMemoryLimits,
) -> Result<Qwen3_5ResidentModel, QuantizationError> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, PROBE_MAXIMUM_OUTPUT_TOKENS)
        .map_err(probe_validation_failure)?;
    let chunking = probe_step(Qwen3_5ModelChunkingConfiguration::new(
        FULL_ATTENTION_KV_GROWTH_TOKENS,
        PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
        SSD_PAGING_PREFILL_LAYER_INTERVAL,
        SSD_PAGING_GENERATION_LAYER_INTERVAL,
    ))?;
    let runtime = mlx_step(
        MlxRuntime::initialize(memory_limits),
        "initialize the probe runtime",
    )?;
    let mut model = probe_step(Qwen3_5ResidentModel::load(
        runtime,
        validated_artifact,
        model_directory,
        chunking,
    ))?;
    probe_step(model.materialize_complete_weights_matching_production())?;
    Ok(model)
}

impl Qwen3_5SensitivityProbe {
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.module_descriptors.len()
    }

    /// KL(reference ‖ this model) at every position of one forward. The
    /// reference logits may come from another model loaded earlier in the
    /// same process: the runtime is process-global, so its arrays outlive
    /// the model that produced them.
    pub fn per_position_kl_divergences(
        &self,
        reference_logits: &MlxArray,
        candidate_logits: &MlxArray,
    ) -> Result<Vec<f64>, QuantizationError> {
        per_position_kl_divergences(self.model.runtime(), reference_logits, candidate_logits)
    }

    #[must_use]
    pub const fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    #[must_use]
    pub fn bfloat16_source_directory(&self) -> Option<&Path> {
        match &self.bfloat16_source {
            Some(bfloat16_source) => Some(&bfloat16_source.directory),
            None => None,
        }
    }

    /// Whether swapping this module must also swap the head: with tied
    /// embeddings the head is a second reference to the embedding arrays,
    /// and a quantized tied embedding drives both, as mlx-lm's tied
    /// embedding does, so probing only the input side would understate it.
    fn ties_head_to(&self, module_name: &str) -> bool {
        self.module_locations.get(module_name) == Some(&ModuleLocation::Embedding)
            && self.model.config.has_tied_embeddings()
    }

    /// The replacement for one probe, by reference path.
    fn replacement_weights(
        &mut self,
        module_name: &str,
        current_weights: &Qwen3_5AffineWeights,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<Qwen3_5AffineWeights, QuantizationError> {
        let runtime = self.model.runtime();
        let Some(bfloat16_source) = &mut self.bfloat16_source else {
            if bit_width.keeps_bfloat16() {
                return mlx_step(current_weights.retained_reference(), "retain the module");
            }
            return quantize_dequantized_bfloat16(runtime, current_weights, bit_width, group_size);
        };
        let source_weight = bfloat16_source.read_module_weight(runtime, module_name)?;
        encoded_from_source(runtime, &source_weight, bit_width, group_size)
    }

    /// Prefills every full chunk but the last, then forwards the last chunk
    /// and returns its logits at every position.
    fn forward_logits(&self, token_ids: &[u32]) -> Result<MlxArray, QuantizationError> {
        let mut decoder_state = mlx_step(
            self.model
                .empty_request_decoder_state_matching_loaded_weights(
                    FULL_ATTENTION_KV_GROWTH_TOKENS as i32,
                ),
            "allocate the probe decoder state",
        )?;
        let chunk_tokens = self.prefill_chunk_tokens.max(1);
        let final_chunk_start = (token_ids.len().saturating_sub(1) / chunk_tokens) * chunk_tokens;
        let mut chunk_start = 0;
        while chunk_start < final_chunk_start {
            let chunk = &token_ids[chunk_start..chunk_start + chunk_tokens];
            probe_step(
                self.model
                    .prefill_chunk(chunk, chunk_start as u32, &mut decoder_state),
            )?;
            chunk_start += chunk_tokens;
        }
        let final_output = probe_step(
            self.model
                .forward_chunk_with_pre_final_normalization_hidden_states(
                    &token_ids[final_chunk_start..],
                    final_chunk_start as u32,
                    &mut decoder_state,
                ),
        )?;
        let logits =
            probe_step(self.model.all_position_vocabulary_logits(
                final_output.pre_final_normalization_hidden_states(),
            ))?;
        mlx_step(logits.evaluate(), "evaluate the probe logits")?;
        Ok(logits)
    }
}

impl Bfloat16Source {
    /// The module's bf16 source weight in its logical shape, lazily.
    fn read_module_weight(
        &mut self,
        runtime: &MlxRuntime,
        module_name: &str,
    ) -> Result<MlxArray, QuantizationError> {
        let Some(descriptor) = self.inventory.resolve(module_name).cloned() else {
            return Err(unresolvable(
                module_name,
                "the bf16 source has no such module",
            ));
        };
        read_module_weight(runtime, &mut self.shard_maps, &descriptor)
    }
}

/// The restore token's engine state: proof the swap came from this probe;
/// the arrays themselves wait in `swapped_out_weights`.
struct Qwen3_5SwapMarker;

impl SensitivityProbeModel for Qwen3_5SensitivityProbe {
    /// Logits at every position of the sample's final chunk, on device, in
    /// the model dtype.
    type Logits = MlxArray;

    fn quantizable_modules(&self) -> Vec<QuantizableModuleDescriptor> {
        self.module_descriptors.clone()
    }

    /// `Some(bits)` for a quantized affine module, `None` for bf16 or an
    /// unresolvable module.
    fn current_bit_width(&self, module_name: &str) -> Option<BitWidth> {
        for descriptor in &self.module_descriptors {
            if descriptor.module_name == module_name {
                let (bits, _) = descriptor.source_quantization?;
                return BitWidth::new(bits).ok();
            }
        }
        None
    }

    fn forward(
        &mut self,
        sample: &CalibrationForwardSample,
    ) -> Result<Self::Logits, QuantizationError> {
        self.forward_logits(&sample.token_ids)
    }

    /// Swaps the module's weights for the probed width, keeping what was
    /// there until the restore. A module the probe cannot reach fails with
    /// `ProbeModelFailure`, which the sweep counts as skipped.
    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError> {
        match self.location_of(module_name)? {
            ModuleLocation::DecoderLayerRouter {
                decoder_layer_index,
            } => self.swap_router(module_name, decoder_layer_index, bit_width, group_size)?,
            ModuleLocation::DecoderLayerRoutedExperts {
                decoder_layer_index,
                projection: projection @ (FeedForwardProjection::Gate | FeedForwardProjection::Up),
            } => self.swap_expert_gate_or_up(
                module_name,
                decoder_layer_index,
                projection,
                bit_width,
                group_size,
            )?,
            _ => self.swap_affine_module(module_name, bit_width, group_size)?,
        }
        Ok(ModuleRestoreToken {
            module_name: module_name.to_owned(),
            engine_state: Box::new(Qwen3_5SwapMarker),
        })
    }

    /// Puts the previous arrays back.
    fn restore_module(
        &mut self,
        restore_token: ModuleRestoreToken,
    ) -> Result<(), QuantizationError> {
        let is_qwen3_5_swap = restore_token.engine_state.is::<Qwen3_5SwapMarker>();
        let previous = self.swapped_out_weights.remove(&restore_token.module_name);
        let (true, Some(previous)) = (is_qwen3_5_swap, previous) else {
            return Err(QuantizationError::ProbeModelFailure {
                description: format!(
                    "restore token for {} did not come from a Qwen3.5 swap",
                    restore_token.module_name
                ),
            });
        };
        self.restore_swapped_out(&restore_token.module_name, previous)?;
        if let Some(SwappedOutWeights::Affine(previous_head)) = self
            .swapped_out_weights
            .remove(&tied_head_key(&restore_token.module_name))
        {
            self.model.base.weights.language_model_head_weights = previous_head;
        }
        Ok(())
    }

    fn kl_divergence(
        &self,
        reference_logits: &Self::Logits,
        probe_logits: &Self::Logits,
    ) -> Result<f64, QuantizationError> {
        mean_kl_divergence(self.model.runtime(), reference_logits, probe_logits)
    }

    /// Drains the allocator cache: large tensors otherwise accumulate in
    /// MLX's page cache and the process is killed within about ten modules.
    fn release_transient_memory(&mut self) -> Result<(), QuantizationError> {
        mlx_step(
            self.model
                .runtime()
                .synchronize_gpu_stream_and_clear_allocator_cache(),
            "drain the allocator cache between probes",
        )
    }
}

/// Any engine failure, as the sweep's typed probe failure.
/// The validation error's own text names only the stage that failed; the
/// public reason carries the config field or file behind it, which is what
/// someone converting a new base needs to see.
fn probe_validation_failure(
    validation_error: crate::Qwen3_5ArtifactValidationError,
) -> QuantizationError {
    QuantizationError::ProbeModelFailure {
        description: validation_error.public_failure_reason(),
    }
}

fn probe_step<Value, Failure: std::fmt::Display>(
    outcome: Result<Value, Failure>,
) -> Result<Value, QuantizationError> {
    match outcome {
        Ok(value) => Ok(value),
        Err(failure) => Err(QuantizationError::ProbeModelFailure {
            description: failure.to_string(),
        }),
    }
}

/// Where a tied head's swapped-out arrays wait beside the embedding's.
fn tied_head_key(module_name: &str) -> String {
    format!("{module_name}#tied-head")
}

fn unresolvable(module_name: &str, reason: &str) -> QuantizationError {
    QuantizationError::ProbeModelFailure {
        description: format!("cannot probe {module_name}: {reason}"),
    }
}
