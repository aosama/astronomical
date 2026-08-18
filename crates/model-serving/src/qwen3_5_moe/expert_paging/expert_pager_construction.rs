//! Startup construction of validated expert layer plans and memory admission.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntime;

use super::expert_pager::{ExpertPagingError, Qwen3_5ExpertPager, Qwen3_5ExpertPagerGeometry};
use super::quantized_expert_layer_plan::build_quantized_expert_layer_plan_with_stored_names_and_header_cache;
use crate::MlxAllocationBudget;
use crate::expert_paging::safetensors_header::SafetensorsHeader;
use crate::expert_paging::{QuantizationMode, QuantizedExpertLayerPlan};
use crate::qwen3_5::{ModelWeightStorage, Qwen3_5Config};

impl Qwen3_5ExpertPager {
    /// Returns the number of MoE layers with validated layer plans.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_plans.len()
    }

    /// Builds layer plans for all MoE layers at startup.
    ///
    /// Reads safetensors headers, validates tensor geometry, and pre-computes
    /// per-expert byte strides. Does NOT load expert weights.
    ///
    /// `weight_map` maps tensor names (e.g.,
    /// `language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight`) to
    /// shard file names (e.g., `model-00001-of-00005.safetensors`).
    /// `model_dir` is the directory containing the shard files.
    ///
    /// `configured_mlx_memory_cap_bytes` is the worker's resolved numeric
    /// admission ceiling for all expert paging operations.
    pub fn new(
        _runtime: &MlxRuntime,
        model_dir: PathBuf,
        weight_map: &HashMap<String, String>,
        stored_tensor_name_by_canonical_name: &HashMap<String, String>,
        config: &Qwen3_5Config,
        configured_mlx_memory_cap_bytes: usize,
        include_mtp_sparse_expert_layer: bool,
    ) -> Result<Self, ExpertPagingError> {
        let decoder_layer_count = config.layer_count() as usize;
        let mut layer_plans =
            Vec::with_capacity(decoder_layer_count + usize::from(include_mtp_sparse_expert_layer));
        // One model-level cache prevents every decoder layer from reparsing the same shard header.
        // The cache owns bounded metadata only and is dropped after all byte-range plans are built.
        let mut safetensors_header_by_source_file = HashMap::<PathBuf, SafetensorsHeader>::new();
        let quantization_mode = match config.model_weight_storage() {
            ModelWeightStorage::NativeBfloat16 => QuantizationMode::NativeBfloat16,
            ModelWeightStorage::AffineQuantized => QuantizationMode::Affine,
        };
        for decoder_layer_index in 0..decoder_layer_count {
            let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
            let layer_plan = build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
                &model_dir,
                weight_map,
                stored_tensor_name_by_canonical_name,
                &layer_prefix,
                config,
                quantization_mode,
                &mut safetensors_header_by_source_file,
            )?;
            layer_plans.push(layer_plan);
        }
        // MTP has a separate tensor namespace but shares the same pager and
        // Rust pager. Appending its one sparse layer gives it a stable index
        // immediately after the target decoder layers without merging artifact
        // inventories during validation.
        if include_mtp_sparse_expert_layer {
            let mtp_layer_plan =
                build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
                    &model_dir,
                    weight_map,
                    stored_tensor_name_by_canonical_name,
                    "language_model.mtp.layers.0.mlp",
                    config,
                    quantization_mode,
                    &mut safetensors_header_by_source_file,
                )?;
            layer_plans.push(mtp_layer_plan);
        }

        // Streaming workspace and temporary decode routes are bounded by complete
        // layer geometry. Track the largest complete layer so budgets and telemetry
        // describe whole-layer ownership rather than one routed top-K page.
        let maximum_expert_page_bytes = layer_plans.iter().try_fold(
            0u64,
            |maximum_expert_page_bytes, layer_plan| -> Result<_, ExpertPagingError> {
                Ok(maximum_expert_page_bytes.max(layer_plan.complete_expert_payload_byte_count()?))
            },
        )?;
        let memory_budget = MlxAllocationBudget::new(
            maximum_expert_page_bytes,
            configured_mlx_memory_cap_bytes as u64,
        );
        let resident_expert_source_files = retain_resident_expert_source_files(&layer_plans)?;
        Ok(Self {
            layer_plans,
            memory_budget,
            resident_expert_source_files,
        })
    }

    /// Adds the independently packaged sparse MTP layer without changing target source ownership.
    ///
    /// Every fallible plan, descriptor, and byte-accounting operation finishes before the pager
    /// is mutated so optional-drafter failure cannot damage the requestable target model.
    pub(crate) fn append_standalone_mtp_layer(
        &mut self,
        model_directory: PathBuf,
        tensor_name_to_shard_file_name: &HashMap<String, String>,
        stored_tensor_name_by_canonical_name: &HashMap<String, String>,
        config: &Qwen3_5Config,
        experts_per_token: usize,
    ) -> Result<Qwen3_5ExpertPagerGeometry, ExpertPagingError> {
        let quantization_mode = match config.model_weight_storage() {
            ModelWeightStorage::NativeBfloat16 => QuantizationMode::NativeBfloat16,
            ModelWeightStorage::AffineQuantized => QuantizationMode::Affine,
        };
        let mut safetensors_header_by_source_file = HashMap::new();
        let standalone_layer_plan =
            build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
                &model_directory,
                tensor_name_to_shard_file_name,
                stored_tensor_name_by_canonical_name,
                "language_model.mtp.layers.0.mlp",
                config,
                quantization_mode,
                &mut safetensors_header_by_source_file,
            )?;
        let standalone_complete_payload_bytes =
            standalone_layer_plan.complete_expert_payload_byte_count()?;
        let complete_expert_payload_bytes = self
            .complete_expert_payload_byte_count()?
            .checked_add(standalone_complete_payload_bytes)
            .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                description: "complete standalone MTP expert payload byte count overflowed"
                    .to_owned(),
            })?;
        let routed_expert_count = experts_per_token.min(standalone_layer_plan.expert_capacity);
        let standalone_routed_payload_bytes = u64::try_from(
            u128::from(standalone_complete_payload_bytes)
                .saturating_mul(routed_expert_count as u128)
                / (standalone_layer_plan.expert_capacity.max(1) as u128),
        )
        .unwrap_or(u64::MAX);
        let largest_routed_expert_page_bytes = self
            .maximum_routed_expert_page_bytes(experts_per_token)?
            .max(standalone_routed_payload_bytes);
        let largest_complete_expert_layer_bytes = self
            .maximum_expert_page_bytes()
            .max(standalone_complete_payload_bytes);
        let standalone_source_files =
            retain_resident_expert_source_files(std::slice::from_ref(&standalone_layer_plan))?;
        let active_memory_ceiling_bytes = self.memory_budget.active_memory_ceiling_bytes();
        let observed_transient_high_water_bytes =
            self.memory_budget.observed_transient_high_water_bytes();
        let replacement_memory_budget = MlxAllocationBudget::new(
            largest_complete_expert_layer_bytes,
            active_memory_ceiling_bytes,
        );

        self.layer_plans.push(standalone_layer_plan);
        self.resident_expert_source_files
            .extend(standalone_source_files);
        self.memory_budget = replacement_memory_budget;
        self.memory_budget
            .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
        Ok(Qwen3_5ExpertPagerGeometry {
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
        })
    }

    /// Removes the optional trailing MTP layer while preserving every target decoder plan.
    pub(crate) fn remove_optional_mtp_layer(
        &mut self,
        target_decoder_layer_count: usize,
        experts_per_token: usize,
    ) -> Result<Qwen3_5ExpertPagerGeometry, ExpertPagingError> {
        if self.layer_plans.len() > target_decoder_layer_count {
            self.layer_plans.truncate(target_decoder_layer_count);
        }
        let complete_expert_payload_bytes = self.complete_expert_payload_byte_count()?;
        let largest_complete_expert_layer_bytes = self.layer_plans.iter().try_fold(
            0_u64,
            |largest_payload_bytes, layer_plan| -> Result<_, ExpertPagingError> {
                Ok(largest_payload_bytes.max(layer_plan.complete_expert_payload_byte_count()?))
            },
        )?;
        let largest_routed_expert_page_bytes =
            self.maximum_routed_expert_page_bytes(experts_per_token)?;
        let active_memory_ceiling_bytes = self.memory_budget.active_memory_ceiling_bytes();
        let observed_transient_high_water_bytes =
            self.memory_budget.observed_transient_high_water_bytes();
        self.memory_budget = MlxAllocationBudget::new(
            largest_complete_expert_layer_bytes,
            active_memory_ceiling_bytes,
        );
        self.memory_budget
            .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
        Ok(Qwen3_5ExpertPagerGeometry {
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
        })
    }
}

fn retain_resident_expert_source_files(
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Vec<(PathBuf, File)>, ExpertPagingError> {
    // Open once while artifact paths are known-valid. BTreeSet deduplicates and
    // stabilizes descriptor order; later promotions clone these open handles and
    // do not depend on path lookup during a later resident promotion.
    layer_plans
        .iter()
        .flat_map(|layer_plan| layer_plan.tensor_sources.iter())
        .map(|tensor_source| tensor_source.source_file.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|source_file_path| {
            File::open(&source_file_path)
                .map(|source_file| (source_file_path.clone(), source_file))
                .map_err(|source| ExpertPagingError::ResidentSourceOpen {
                    source_file: source_file_path,
                    source,
                })
        })
        .collect()
}
