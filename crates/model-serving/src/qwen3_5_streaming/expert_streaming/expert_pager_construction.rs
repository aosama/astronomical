//! Startup construction of validated expert layer plans and memory admission.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use astronomical_runtime_integration::{MlxRuntime, PositionalFileReadMetrics};

use super::expert_pager::{ExpertPagingError, Qwen3_5ExpertPager};
use crate::MlxAllocationAdmission;
use crate::expert_paging::QuantizedExpertLayerPlan;
use crate::expert_paging::safetensors_header::SafetensorsHeader;
use crate::memory::ExpertStreamingSessionCounters;
use crate::qwen3_5_core::artifacts::quantized_expert_layer_plan;
use crate::qwen3_5_core::configuration::Qwen3_5Config;

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
    ) -> Result<Self, ExpertPagingError> {
        let decoder_layer_count = config.layer_count() as usize;
        let layer_plans = self_shard_layer_plans(
            decoder_layer_count,
            &model_dir,
            weight_map,
            stored_tensor_name_by_canonical_name,
            config,
        )?;

        // Streaming workspace and temporary decode routes are bounded by complete
        // layer geometry: the packed source ownership plus the decoded
        // bfloat16 output a multi-token forward materializes for the layer.
        let maximum_expert_page_bytes = layer_plans.iter().try_fold(
            0u64,
            |maximum_expert_page_bytes, layer_plan| -> Result<_, ExpertPagingError> {
                let complete_source_payload_bytes =
                    layer_plan.complete_expert_payload_byte_count()?;
                let decoded_codebook_weight_shapes =
                    layer_plan.jangtq2_decoded_weight_page_shapes(layer_plan.expert_capacity)?;
                let complete_page_peak_allocation_bytes =
                    crate::memory::expert_page_peak_allocation_bytes(
                        complete_source_payload_bytes,
                        &decoded_codebook_weight_shapes,
                        size_of::<u16>(),
                    )
                    .ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                        description: "complete expert-page allocation size overflowed".to_owned(),
                    })?;
                Ok(maximum_expert_page_bytes.max(complete_page_peak_allocation_bytes))
            },
        )?;
        let memory_budget = MlxAllocationAdmission::new(
            maximum_expert_page_bytes,
            configured_mlx_memory_cap_bytes as u64,
        );
        let resident_expert_source_files = retain_resident_expert_source_files(&layer_plans)?;
        Ok(Self {
            layer_plans,
            memory_budget,
            resident_expert_source_files,
            // Conservative until the model proves the fused kernel: every
            // page prices its decode peak.
            decode_tokens_multiply_packed_pages: std::cell::Cell::new(false),
            session_counters: RefCell::new(ExpertStreamingSessionCounters::default()),
            session_read_metrics: Arc::new(PositionalFileReadMetrics::default()),
            session_process_io_baseline: RefCell::new(None),
        })
    }
}

/// Builds the shard-backed layer plans for the MoE decoder layers from the
/// validated safetensors index.
fn self_shard_layer_plans(
    decoder_layer_count: usize,
    model_dir: &Path,
    weight_map: &HashMap<String, String>,
    stored_tensor_name_by_canonical_name: &HashMap<String, String>,
    config: &Qwen3_5Config,
) -> Result<Vec<QuantizedExpertLayerPlan>, ExpertPagingError> {
    let mut layer_plans = Vec::with_capacity(decoder_layer_count);
    // One model-level cache prevents every decoder layer from reparsing the same shard header.
    // The cache owns bounded metadata only and is dropped after all byte-range plans are built.
    let mut safetensors_header_by_source_file = HashMap::<PathBuf, SafetensorsHeader>::new();
    for decoder_layer_index in 0..decoder_layer_count {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.mlp");
        let layer_plan = quantized_expert_layer_plan::build_quantized_expert_layer_plan_with_stored_names_and_header_cache(
            model_dir,
            weight_map,
            stored_tensor_name_by_canonical_name,
            &layer_prefix,
            config,
            &mut safetensors_header_by_source_file,
        )?;
        layer_plans.push(layer_plan);
    }
    Ok(layer_plans)
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
