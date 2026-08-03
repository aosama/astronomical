//! Startup construction of validated expert layer plans and memory admission.

use std::collections::HashMap;
use std::path::PathBuf;

use astronomical_ipc_protocol::ExpertStorageFormat;

use super::aligned_expert_pack_loader::discover_aligned_expert_pack_layers;
use super::complete_layer_retention::complete_layer_expert_payload_byte_count;
use super::expert_pager::{ExpertPager, ExpertPagingError};
use super::memory_budget::LiveMetalBudget;
use super::quantized_expert_layer_plan::build_quantized_expert_layer_plan;
use super::quantized_expert_manifest::QuantizationMode;
use crate::qwen3_5::{ModelWeightStorage, Qwen3_5Config};

impl ExpertPager {
    /// Returns the expert file layout selected during complete-revision discovery.
    #[must_use]
    pub fn expert_storage_format(&self) -> ExpertStorageFormat {
        if self.decoder_layer_plan_count > 0
            && self.aligned_expert_pack_layer_count() == self.decoder_layer_plan_count
        {
            ExpertStorageFormat::AstronomicalAligned
        } else {
            ExpertStorageFormat::StandardSafetensors
        }
    }

    /// Returns the number of MoE layers with validated layer plans.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_plans.len()
    }

    /// Returns the number of layers activated through the complete prepared revision.
    #[must_use]
    pub fn aligned_expert_pack_layer_count(&self) -> usize {
        self.aligned_expert_pack_layers
            .iter()
            .filter(|aligned_expert_pack_layer| aligned_expert_pack_layer.is_some())
            .count()
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
        model_dir: PathBuf,
        model_id: &str,
        model_revision: &str,
        weight_map: &HashMap<String, String>,
        config: &Qwen3_5Config,
        configured_mlx_memory_cap_bytes: usize,
        include_mtp_sparse_expert_layer: bool,
    ) -> Result<Self, ExpertPagingError> {
        let decoder_layer_count = config.layer_count() as usize;
        let mut layer_plans =
            Vec::with_capacity(decoder_layer_count + usize::from(include_mtp_sparse_expert_layer));
        let quantization_mode = match config.model_weight_storage() {
            ModelWeightStorage::NativeBfloat16 => QuantizationMode::NativeBfloat16,
            ModelWeightStorage::AffineQuantized => QuantizationMode::Affine,
        };
        for decoder_layer_index in 0..decoder_layer_count {
            let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
            let layer_plan = build_quantized_expert_layer_plan(
                &model_dir,
                weight_map,
                &layer_prefix,
                config,
                quantization_mode,
            )?;
            layer_plans.push(layer_plan);
        }
        if include_mtp_sparse_expert_layer {
            let mtp_layer_plan = build_quantized_expert_layer_plan(
                &model_dir,
                weight_map,
                "language_model.mtp.layers.0.mlp",
                config,
                quantization_mode,
            )?;
            layer_plans.push(mtp_layer_plan);
        }

        let mut aligned_expert_pack_layers = discover_aligned_expert_pack_layers(
            &model_dir,
            model_id,
            model_revision,
            &layer_plans[..decoder_layer_count],
        );
        aligned_expert_pack_layers.resize_with(layer_plans.len(), || None);
        let mut maximum_expert_page_bytes = 0;
        for (layer_index, layer_plan) in layer_plans.iter().enumerate() {
            maximum_expert_page_bytes = maximum_expert_page_bytes.max(
                complete_layer_expert_payload_byte_count(layer_plan, layer_index)?,
            );
        }
        let memory_budget = LiveMetalBudget::new(
            maximum_expert_page_bytes,
            configured_mlx_memory_cap_bytes as u64,
        );
        Ok(Self {
            layer_plans,
            decoder_layer_plan_count: decoder_layer_count,
            aligned_expert_pack_layers,
            memory_budget,
        })
    }
}
