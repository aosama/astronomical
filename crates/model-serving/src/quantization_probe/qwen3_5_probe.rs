//! The Qwen3.5 probe: one resident model whose modules can be swapped.
//!
//! The swap-forward-restore cycle is ported from OptiQ's `_exact_with_bf16_reference` (`optiq/core/sensitivity.py`); the model side is Astronomical's own Qwen3.5 engine.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_quantization::module_inventory::{
    ModuleInventory, ModuleWeightSource, QuantizableModuleDescriptor,
};
use astronomical_quantization::sensitivity::{
    CalibrationForwardSample, ModuleRestoreToken, SensitivityProbeModel,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};
use mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_resident::Qwen3_5ResidentModel;
use crate::{Qwen3_5ArtifactValidator, Qwen3_5ModelChunkingConfiguration};

use super::module_locations::{ModuleLocation, affine_weights_at_mut, module_location_for};
use super::simulated_quantization::{
    encoded_from_source, mean_kl_divergence, mlx_step, quantize_dequantized_bfloat16,
};

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
    swapped_out_weights: BTreeMap<String, Qwen3_5AffineWeights>,
}

/// The bf16 source a uniform-baseline probe re-encodes modules from.
struct Bfloat16Source {
    directory: PathBuf,
    inventory: ModuleInventory,
    open_shards: BTreeMap<String, MlxSafetensors>,
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
            open_shards: BTreeMap::new(),
        }),
        None => None,
    };
    let model = load_resident_model(model_directory, memory_limits)?;
    let mut module_descriptors = Vec::new();
    let mut module_locations = BTreeMap::new();
    for descriptor in inventory.modules() {
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
    let validated_artifact = probe_step(
        Qwen3_5ArtifactValidator::new().validate(model_directory, PROBE_MAXIMUM_OUTPUT_TOKENS),
    )?;
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

    /// The affine weights a module name resolves to, mutably.
    fn affine_weights_mut(
        &mut self,
        module_name: &str,
    ) -> Result<&mut Qwen3_5AffineWeights, QuantizationError> {
        let Some(location) = self.module_locations.get(module_name).copied() else {
            return Err(unresolvable(
                module_name,
                "it has no location in the Qwen3.5 weights",
            ));
        };
        match affine_weights_at_mut(&mut self.model.base.weights, location) {
            Some(affine_weights) => Ok(affine_weights),
            None => Err(unresolvable(
                module_name,
                "routed-expert swaps are not supported by this probe yet",
            )),
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
    /// The module's bf16 source tensor, lazily from its shard. Only whole
    /// tensors are streamed; fused halves and per-expert stacks belong to
    /// routed experts, which this probe does not swap.
    fn read_module_weight(
        &mut self,
        runtime: &MlxRuntime,
        module_name: &str,
    ) -> Result<MlxArray, QuantizationError> {
        let Some(descriptor) = self.inventory.resolve(module_name) else {
            return Err(unresolvable(
                module_name,
                "the bf16 source has no such module",
            ));
        };
        let ModuleWeightSource::Whole {
            shard_file_name,
            tensor_name,
        } = descriptor.weight_source.clone()
        else {
            return Err(unresolvable(
                module_name,
                "its bf16 source is not one whole tensor",
            ));
        };
        if !self.open_shards.contains_key(&shard_file_name) {
            let shard_path = self.directory.join(&shard_file_name);
            let shard_file = mlx_step(File::open(&shard_path), "open a bf16 source shard")?;
            let shard_map = mlx_step(
                runtime.load_safetensors(shard_file, None),
                "map a bf16 source shard",
            )?;
            self.open_shards.insert(shard_file_name.clone(), shard_map);
        }
        mlx_step(
            self.open_shards[&shard_file_name].tensor(&tensor_name),
            "read a bf16 source tensor",
        )
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

    /// Swaps the module's affine weights for the probed width, keeping the
    /// previous arrays in the token. A module the probe cannot reach fails
    /// with `ProbeModelFailure`, which the sweep counts as skipped.
    fn simulate_quantize_module(
        &mut self,
        module_name: &str,
        bit_width: BitWidth,
        group_size: GroupSize,
    ) -> Result<ModuleRestoreToken, QuantizationError> {
        let current_weights = mlx_step(
            self.affine_weights_mut(module_name)?.retained_reference(),
            "retain the module",
        )?;
        let replacement =
            self.replacement_weights(module_name, &current_weights, bit_width, group_size)?;
        let previous = std::mem::replace(self.affine_weights_mut(module_name)?, replacement);
        self.swapped_out_weights
            .insert(module_name.to_owned(), previous);
        if self.ties_head_to(module_name) {
            let head_replacement = mlx_step(
                self.affine_weights_mut(module_name)?.retained_reference(),
                "retain the tied head",
            )?;
            let previous_head = std::mem::replace(
                &mut self.model.base.weights.language_model_head_weights,
                head_replacement,
            );
            self.swapped_out_weights
                .insert(tied_head_key(module_name), previous_head);
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
        *self.affine_weights_mut(&restore_token.module_name)? = previous;
        if let Some(previous_head) = self
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
