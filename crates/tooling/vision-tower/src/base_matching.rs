//! Proves whether a vision tower belongs with a text model.
//!
//! A tower only works with the text weights it was trained beside, and nothing in
//! a config says whether that is true: two models can have matching shapes and
//! unrelated weights. So this compares real tensors. Quantization leaves small
//! tensors such as norms and DeltaNet gates in full precision, and those survive
//! unchanged from the base, which makes them a fingerprint that needs only a few
//! megabytes to check.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::artifact_source::ArtifactSource;
use crate::error::VisionTowerError;
use crate::model_config::{ModelConfigSummary, read_model_config};
use crate::tensor_agreement::{TensorAgreement, TensorSide, classify_tensor_pair};
use crate::tensor_names::{
    canonical_tensor_name, is_quantization_companion, is_vision_tower_tensor,
    scales_name_for_weight,
};
use crate::tensor_values::is_float_data_type;
use crate::tower_inspection::tower_tensor_count;
use crate::weight_inventory::{LocatedTensor, WeightInventory, read_weight_inventory};

// Spread across the model, this is enough for a confident answer at about 4 MiB
// a tensor at most, and it keeps the check to a few dozen small requests.
const MAXIMUM_COMPARED_TENSORS: usize = 24;
const MAXIMUM_COMPARED_TENSOR_BYTES: u64 = 4 * 1024 * 1024;
// One or two matching tensors could be coincidence or a frozen constant.
const MINIMUM_INFORMATIVE_TENSORS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BaseMatchVerdict {
    /// Every informative tensor agrees: built from the same weights.
    SameBase,
    /// Many agree and some differ: an edited model, which needs image tests.
    DerivedFromBase,
    /// The shapes fit, but the weights were trained separately.
    NoSharedWeights,
    /// They cannot work together, for the reasons listed.
    Incompatible,
    /// Too little to compare to say either way.
    Inconclusive,
}

impl BaseMatchVerdict {
    /// Whether adding this tower to the text model can reasonably proceed.
    #[must_use]
    pub const fn allows_adding(self) -> bool {
        matches!(self, Self::SameBase | Self::DerivedFromBase)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ComparedTensor {
    pub name: String,
    pub agreement: TensorAgreement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BaseMatchReport {
    pub text_artifact: String,
    pub text_revision: Option<String>,
    pub tower_artifact: String,
    pub tower_revision: Option<String>,
    pub verdict: BaseMatchVerdict,
    /// Why the pair is incompatible; empty otherwise.
    pub incompatibilities: Vec<String>,
    /// Checks that could not run because a config did not state the value.
    pub unchecked: Vec<String>,
    pub text_already_has_tower: bool,
    pub comparable_tensor_count: u64,
    pub compared_tensors: Vec<ComparedTensor>,
}

/// Told after each comparison, so a slow connection stays visible.
pub trait BaseMatchProgress {
    fn tensor_compared(&mut self, completed_count: usize, total_count: usize);
}

pub async fn match_tower_to_text(
    text_source: &ArtifactSource,
    tower_source: &ArtifactSource,
    progress: &mut dyn BaseMatchProgress,
) -> Result<BaseMatchReport, VisionTowerError> {
    let text_config = read_model_config(text_source).await?;
    let tower_config = read_model_config(tower_source).await?;
    let text_inventory = read_weight_inventory(text_source).await?;
    let tower_inventory = read_weight_inventory(tower_source).await?;
    let text_listing = text_source.listing();
    let tower_listing = tower_source.listing();
    let mut report = BaseMatchReport {
        text_artifact: text_listing.display_name.clone(),
        text_revision: text_listing.pinned_revision.clone(),
        tower_artifact: tower_listing.display_name.clone(),
        tower_revision: tower_listing.pinned_revision.clone(),
        verdict: BaseMatchVerdict::Inconclusive,
        incompatibilities: Vec::new(),
        unchecked: Vec::new(),
        text_already_has_tower: tower_tensor_count(&text_inventory) > 0,
        comparable_tensor_count: 0,
        compared_tensors: Vec::new(),
    };
    check_configs(&text_config, &tower_config, &mut report);
    if tower_tensor_count(&tower_inventory) == 0 {
        report
            .incompatibilities
            .push(format!("{} has no vision tower", report.tower_artifact));
    }
    if !report.incompatibilities.is_empty() {
        report.verdict = BaseMatchVerdict::Incompatible;
        return Ok(report);
    }
    let comparable_pairs = comparable_tensor_pairs(&text_inventory, &tower_inventory);
    report.comparable_tensor_count = comparable_pairs.len() as u64;
    let selected_pairs = evenly_spaced_pairs(&comparable_pairs, MAXIMUM_COMPARED_TENSORS);
    for (completed_index, tensor_pair) in selected_pairs.iter().enumerate() {
        let agreement = compare_pair(text_source, tower_source, tensor_pair).await?;
        report.compared_tensors.push(ComparedTensor {
            name: tensor_pair.canonical_name.to_owned(),
            agreement,
        });
        progress.tensor_compared(completed_index + 1, selected_pairs.len());
    }
    report.verdict = verdict_from_comparisons(&report.compared_tensors);
    Ok(report)
}

fn check_configs(
    text_config: &ModelConfigSummary,
    tower_config: &ModelConfigSummary,
    report: &mut BaseMatchReport,
) {
    let tower_output_width = match &tower_config.vision {
        Some(vision_config) => vision_config.output_hidden_size,
        None => None,
    };
    match (tower_output_width, text_config.text_hidden_size) {
        (Some(tower_width), Some(text_width)) if tower_width != text_width => {
            report.incompatibilities.push(format!(
                "the tower outputs {tower_width} channels and the text model takes {text_width}"
            ));
        }
        (Some(_), Some(_)) => {}
        _ => report.unchecked.push("tower output width".to_owned()),
    }
    let token_fields = [
        (
            "image_token_id",
            text_config.image_token_id,
            tower_config.image_token_id,
        ),
        (
            "video_token_id",
            text_config.video_token_id,
            tower_config.video_token_id,
        ),
        (
            "vision_start_token_id",
            text_config.vision_start_token_id,
            tower_config.vision_start_token_id,
        ),
        (
            "vision_end_token_id",
            text_config.vision_end_token_id,
            tower_config.vision_end_token_id,
        ),
    ];
    for (field_name, text_token, tower_token) in token_fields {
        match (text_token, tower_token) {
            (Some(text_token), Some(tower_token)) if text_token != tower_token => {
                report.incompatibilities.push(format!(
                    "{field_name} is {text_token} in the text model and {tower_token} beside the tower"
                ));
            }
            (Some(_), Some(_)) => {}
            _ => report.unchecked.push(field_name.to_owned()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TensorPair<'inventory> {
    canonical_name: &'inventory str,
    text_tensor: &'inventory LocatedTensor,
    tower_side_tensor: &'inventory LocatedTensor,
}

/// Text tensors held unquantized, at one shape, on both sides, in a stable order.
fn comparable_tensor_pairs<'inventory>(
    text_inventory: &'inventory WeightInventory,
    tower_inventory: &'inventory WeightInventory,
) -> Vec<TensorPair<'inventory>> {
    let text_tensors = comparable_tensors_by_canonical_name(text_inventory);
    let tower_side_tensors = comparable_tensors_by_canonical_name(tower_inventory);
    let mut comparable_pairs = Vec::new();
    for (canonical_name, text_tensor) in text_tensors {
        let Some(tower_side_tensor) = tower_side_tensors.get(canonical_name) else {
            continue;
        };
        if text_tensor.entry.shape == tower_side_tensor.entry.shape {
            comparable_pairs.push(TensorPair {
                canonical_name,
                text_tensor,
                tower_side_tensor,
            });
        }
    }
    comparable_pairs
}

fn comparable_tensors_by_canonical_name(
    inventory: &WeightInventory,
) -> BTreeMap<&str, &LocatedTensor> {
    let mut comparable_tensors = BTreeMap::new();
    for (tensor_name, located_tensor) in &inventory.tensors {
        if is_comparable_text_tensor(inventory, tensor_name, located_tensor) {
            comparable_tensors.insert(canonical_tensor_name(tensor_name), located_tensor);
        }
    }
    comparable_tensors
}

fn is_comparable_text_tensor(
    inventory: &WeightInventory,
    tensor_name: &str,
    located_tensor: &LocatedTensor,
) -> bool {
    if is_vision_tower_tensor(tensor_name)
        || is_quantization_companion(tensor_name)
        || !is_float_data_type(&located_tensor.entry.data_type)
        || located_tensor.entry.byte_count == 0
        || located_tensor.entry.byte_count > MAXIMUM_COMPARED_TENSOR_BYTES
    {
        return false;
    }
    match scales_name_for_weight(tensor_name) {
        Some(scales_name) => !inventory.tensors.contains_key(&scales_name),
        None => true,
    }
}

/// Spread across the model, so one edited stretch of layers cannot decide alone.
fn evenly_spaced_pairs<'inventory>(
    comparable_pairs: &[TensorPair<'inventory>],
    maximum_count: usize,
) -> Vec<TensorPair<'inventory>> {
    let selected_count = comparable_pairs.len().min(maximum_count);
    let mut selected_pairs = Vec::with_capacity(selected_count);
    for selection_index in 0..selected_count {
        let pair_index = selection_index * comparable_pairs.len() / selected_count;
        selected_pairs.push(comparable_pairs[pair_index]);
    }
    selected_pairs
}

async fn compare_pair(
    text_source: &ArtifactSource,
    tower_source: &ArtifactSource,
    tensor_pair: &TensorPair<'_>,
) -> Result<TensorAgreement, VisionTowerError> {
    let text_bytes = read_tensor(text_source, tensor_pair.text_tensor).await?;
    let tower_side_bytes = read_tensor(tower_source, tensor_pair.tower_side_tensor).await?;
    Ok(classify_tensor_pair(
        tensor_pair.canonical_name,
        TensorSide {
            data_type: &tensor_pair.text_tensor.entry.data_type,
            tensor_bytes: &text_bytes,
        },
        TensorSide {
            data_type: &tensor_pair.tower_side_tensor.entry.data_type,
            tensor_bytes: &tower_side_bytes,
        },
    ))
}

async fn read_tensor(
    source: &ArtifactSource,
    located_tensor: &LocatedTensor,
) -> Result<Vec<u8>, VisionTowerError> {
    source
        .read_byte_range(
            &located_tensor.relative_path,
            located_tensor.entry.data_start,
            located_tensor.entry.byte_count,
        )
        .await
}

fn verdict_from_comparisons(compared_tensors: &[ComparedTensor]) -> BaseMatchVerdict {
    let mut shared_count = 0_usize;
    let mut different_count = 0_usize;
    for compared_tensor in compared_tensors {
        if compared_tensor.agreement.is_shared() {
            shared_count += 1;
        } else if compared_tensor.agreement == TensorAgreement::Different {
            different_count += 1;
        }
    }
    let informative_count = shared_count + different_count;
    if informative_count < MINIMUM_INFORMATIVE_TENSORS {
        return BaseMatchVerdict::Inconclusive;
    }
    if different_count == 0 {
        return BaseMatchVerdict::SameBase;
    }
    // Abliteration edits a few directions and leaves most tensors alone, while
    // separately trained models share almost nothing, so a quarter separates them.
    if shared_count * 4 >= informative_count {
        return BaseMatchVerdict::DerivedFromBase;
    }
    BaseMatchVerdict::NoSharedWeights
}

#[cfg(test)]
#[path = "base_matching_tests.rs"]
mod tests;
