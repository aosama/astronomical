//! Operation-local Laguna expert page loaded through the family-neutral reader.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::expert_paging::{
    ExpertWeightPage, QuantizationMode, QuantizedExpertPageManifest, QuantizedExpertShardManifest,
    load_quantized_expert_page,
};
use crate::laguna::model::{LagunaBoundLinear, LagunaExecutionError};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaPagingError;
use super::layer_plan::LagunaSparseLayerPagingPlan;

/// One streamed complete-layer or routed expert page ready for gathered SwiGLU.
#[derive(Debug)]
pub struct LagunaExpertWeightPage {
    manifest: QuantizedExpertPageManifest,
    gate_up: LagunaGateUpPage,
    down: LagunaBoundLinear,
}

/// Mutually exclusive split or fused gate/up ownership for one expert page.
#[derive(Debug)]
enum LagunaGateUpPage {
    Split {
        gate: LagunaBoundLinear,
        up: LagunaBoundLinear,
    },
    Fused(LagunaBoundLinear),
}

impl LagunaExpertWeightPage {
    /// Returns the compact page manifest used to load these arrays.
    #[must_use]
    pub const fn manifest(&self) -> &QuantizedExpertPageManifest {
        &self.manifest
    }

    pub(in crate::laguna) const fn split_gate_up(
        &self,
    ) -> Option<(&LagunaBoundLinear, &LagunaBoundLinear)> {
        match &self.gate_up {
            LagunaGateUpPage::Split { gate, up } => Some((gate, up)),
            LagunaGateUpPage::Fused(_) => None,
        }
    }

    pub(in crate::laguna) const fn down(&self) -> &LagunaBoundLinear {
        &self.down
    }

    pub(in crate::laguna) const fn fused_gate_up(&self) -> Option<&LagunaBoundLinear> {
        match &self.gate_up {
            LagunaGateUpPage::Split { .. } => None,
            LagunaGateUpPage::Fused(fused_gate_up) => Some(fused_gate_up),
        }
    }

    /// Returns physical bytes retained by the page's mutually exclusive projections.
    #[must_use]
    pub fn materialized_payload_byte_count(&self) -> u64 {
        let gate_up_payload_bytes = match &self.gate_up {
            LagunaGateUpPage::Split { gate, up } => gate
                .payload_byte_count()
                .saturating_add(up.payload_byte_count()),
            LagunaGateUpPage::Fused(fused_gate_up) => fused_gate_up.payload_byte_count(),
        };
        gate_up_payload_bytes.saturating_add(self.down.payload_byte_count())
    }
}

impl ExpertWeightPage for LagunaExpertWeightPage {
    fn resident_payload_byte_count(&self) -> u64 {
        self.materialized_payload_byte_count()
    }
}

/// Streams one Laguna expert page through bounded SafeTensors reads.
pub fn load_laguna_expert_page(
    runtime: &MlxRuntime,
    layer_plan: &LagunaSparseLayerPagingPlan,
    expert_ids: &[usize],
    performance_attribution: &mut PerformanceAttribution,
) -> Result<LagunaExpertWeightPage, LagunaPagingError> {
    let materialization_operation = if expert_ids.len() == layer_plan.expert_capacity() {
        PerformanceOperation::MandatoryPrefillCompleteLayerMaterializationWait
    } else {
        PerformanceOperation::MandatoryDecodeRoutePageMaterializationWait
    };
    performance_attribution.measure_operation(
        materialization_operation,
        |performance_attribution| {
            performance_attribution.measure_operation(
                PerformanceOperation::RustExpertStreamingLayerPreparation,
                |_| load_laguna_expert_page_inner(runtime, layer_plan, expert_ids),
            )
        },
    )
}

fn load_laguna_expert_page_inner(
    runtime: &MlxRuntime,
    layer_plan: &LagunaSparseLayerPagingPlan,
    expert_ids: &[usize],
) -> Result<LagunaExpertWeightPage, LagunaPagingError> {
    let manifest = layer_plan.routed_page(expert_ids)?;
    let mut loaded_tensors = load_page_tensors(runtime, &manifest)?;
    let gate = take_projection(&mut loaded_tensors, layer_plan, "gate_proj")?;
    let up = take_projection(&mut loaded_tensors, layer_plan, "up_proj")?;
    let down = take_projection(&mut loaded_tensors, layer_plan, "down_proj")?;
    let fused_gate_up = LagunaBoundLinear::fuse_matching_affine_output_rows(runtime, &gate, &up)
        .map_err(|_| LagunaPagingError::PageExecution {
            description: "affine gate/up fusion failed on a streamed Laguna page",
        })?;
    let gate_up = if let Some(fused_gate_up) = fused_gate_up {
        // Concatenation is lazy in MLX. Evaluate the replacement before dropping
        // split gate/up owners so retained pages cannot keep both representations.
        fused_gate_up.materialize_storage(runtime).map_err(|_| {
            LagunaPagingError::PageExecution {
                description: "fused affine gate/up storage could not materialize",
            }
        })?;
        LagunaGateUpPage::Fused(fused_gate_up)
    } else {
        LagunaGateUpPage::Split { gate, up }
    };
    Ok(LagunaExpertWeightPage {
        manifest,
        gate_up,
        down,
    })
}

fn load_page_tensors(
    runtime: &MlxRuntime,
    manifest: &QuantizedExpertPageManifest,
) -> Result<HashMap<String, MlxArray>, LagunaPagingError> {
    if !page_repeats_tensor_names(manifest) {
        return Ok(load_quantized_expert_page(runtime, manifest, None)?);
    }
    // Per-expert sources can split one compact name across shards. The neutral
    // reader rejects duplicate names, so each shard is loaded alone and stacked.
    let mut fragments_by_name: HashMap<String, Vec<(usize, MlxArray)>> = HashMap::new();
    for (shard_index, source_manifest) in manifest.source_manifests.iter().enumerate() {
        let renamed_manifest = rename_shard_manifest(source_manifest, shard_index);
        let isolated_page = QuantizedExpertPageManifest {
            expert_ids: manifest.expert_ids.clone(),
            page_slot_by_global_expert_id: manifest.page_slot_by_global_expert_id.clone(),
            source_manifests: vec![renamed_manifest],
            payload_byte_count: source_manifest.payload_byte_count,
        };
        let shard_tensors = load_quantized_expert_page(runtime, &isolated_page, None)?;
        let first_expert_id = source_manifest
            .source_intervals
            .iter()
            .map(|interval| interval.expert_start)
            .min()
            .unwrap_or(0);
        for (tensor_name, tensor) in shard_tensors {
            let original_name = original_tensor_name(&tensor_name, shard_index);
            fragments_by_name
                .entry(original_name)
                .or_default()
                .push((first_expert_id, tensor));
        }
    }
    concatenate_fragments(runtime, fragments_by_name)
}

fn page_repeats_tensor_names(manifest: &QuantizedExpertPageManifest) -> bool {
    let mut observed_names = std::collections::BTreeSet::new();
    for source_manifest in &manifest.source_manifests {
        for tensor_range in &source_manifest.tensor_ranges {
            if !observed_names.insert(tensor_range.tensor_name.as_str()) {
                return true;
            }
        }
    }
    false
}

fn rename_shard_manifest(
    source_manifest: &QuantizedExpertShardManifest,
    shard_index: usize,
) -> QuantizedExpertShardManifest {
    let mut renamed = source_manifest.clone();
    for tensor_range in &mut renamed.tensor_ranges {
        tensor_range.tensor_name = format!("{}#{shard_index}", tensor_range.tensor_name);
    }
    for source_interval in &mut renamed.source_intervals {
        source_interval.tensor_name = format!("{}#{shard_index}", source_interval.tensor_name);
    }
    renamed
}

fn original_tensor_name(renamed_tensor_name: &str, shard_index: usize) -> String {
    renamed_tensor_name
        .strip_suffix(&format!("#{shard_index}"))
        .unwrap_or(renamed_tensor_name)
        .to_owned()
}

fn concatenate_fragments(
    runtime: &MlxRuntime,
    mut fragments_by_name: HashMap<String, Vec<(usize, MlxArray)>>,
) -> Result<HashMap<String, MlxArray>, LagunaPagingError> {
    let mut concatenated_tensors = HashMap::new();
    for (tensor_name, mut fragments) in fragments_by_name.drain() {
        fragments.sort_by_key(|(first_expert_id, _)| *first_expert_id);
        let fragment_refs = fragments
            .iter()
            .map(|(_, tensor)| tensor)
            .collect::<Vec<_>>();
        let concatenated = if fragment_refs.len() == 1 {
            fragments.remove(0).1
        } else {
            runtime.concatenate_axis(&fragment_refs, 0).map_err(|_| {
                LagunaPagingError::PageExecution {
                    description: "failed to concatenate per-expert Laguna page fragments",
                }
            })?
        };
        concatenated_tensors.insert(tensor_name, concatenated);
    }
    Ok(concatenated_tensors)
}

fn take_projection(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    layer_plan: &LagunaSparseLayerPagingPlan,
    projection_name: &str,
) -> Result<LagunaBoundLinear, LagunaPagingError> {
    let weight = take_tensor(loaded_tensors, &format!("{projection_name}.weight"))?;
    if layer_plan.quantization_mode() == QuantizationMode::NativeBfloat16 {
        return Ok(LagunaBoundLinear::Native { weight });
    }
    let (bits, group_size) = layer_plan.affine_profile_for_projection(projection_name);
    Ok(LagunaBoundLinear::Affine {
        packed_weight: weight,
        scales: take_tensor(loaded_tensors, &format!("{projection_name}.scales"))?,
        biases: take_tensor(loaded_tensors, &format!("{projection_name}.biases"))?,
        bits,
        group_size,
    })
}

fn take_tensor(
    loaded_tensors: &mut HashMap<String, MlxArray>,
    tensor_name: &str,
) -> Result<MlxArray, LagunaPagingError> {
    loaded_tensors
        .remove(tensor_name)
        .ok_or_else(|| LagunaPagingError::MissingPagedTensor {
            tensor_name: tensor_name.to_owned(),
        })
}

impl From<LagunaExecutionError> for LagunaPagingError {
    fn from(_error: LagunaExecutionError) -> Self {
        Self::PageExecution {
            description: "streamed Laguna page execution failed",
        }
    }
}
