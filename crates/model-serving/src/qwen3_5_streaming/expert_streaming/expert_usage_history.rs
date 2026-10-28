//! Persistent per-model expert routing heat (colibri's `.coli_usage` analog).
//!
//! Decode's first touch of an expert is always a storage miss, so a session
//! starts cold no matter how large the warm tables are. This store persists
//! the per-layer expert demand counts the cache already records during
//! forwards, beside the model artifact, and reloads them at engine
//! construction so historically hot experts are admitted warm and ranked
//! ahead of noise from the first token.
//!
//! The file is derived from the runtime model directory, never a configured
//! or hardwired path. Every failure is non-fatal: an unreadable or stale file
//! leaves the session exactly as cold as it would have been without it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const EXPERT_USAGE_FILE_NAME: &str = ".astronomical-expert-usage.json";
const EXPERT_USAGE_FORMAT_VERSION: u32 = 1;
/// Per-layer cap on persisted entries: routing is sparse, and the cap keeps
/// the file bounded for artifacts with thousands of experts per layer.
const MAXIMUM_PERSISTED_EXPERTS_PER_LAYER: usize = 4_096;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExpertUsageFile {
    format_version: u32,
    model_id: Option<String>,
    layer_count: usize,
    /// Sparse per-layer demand counts keyed by expert id.
    layers: Vec<ExpertUsageLayer>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExpertUsageLayer {
    layer_index: usize,
    demand_counts_by_expert_id: BTreeMap<usize, u64>,
}

/// Resolves the usage-history file inside a model directory.
#[must_use]
pub(crate) fn expert_usage_file_path(model_directory: &Path) -> PathBuf {
    model_directory.join(EXPERT_USAGE_FILE_NAME)
}

/// Atomically writes the sparse demand counts; returns whether the file landed.
/// A failed write logs and returns false — serving must never fail on history.
pub(crate) fn write_expert_usage(
    usage_file_path: &Path,
    model_id: Option<&str>,
    expert_demand_counts_by_layer: &[Vec<u64>],
) -> bool {
    let layers = expert_demand_counts_by_layer
        .iter()
        .enumerate()
        .map(|(layer_index, layer_demand_counts)| {
            persisted_expert_usage_layer(layer_index, layer_demand_counts)
        })
        .collect::<Vec<_>>();
    let usage_file = ExpertUsageFile {
        format_version: EXPERT_USAGE_FORMAT_VERSION,
        model_id: model_id.map(str::to_owned),
        layer_count: expert_demand_counts_by_layer.len(),
        layers,
    };
    let temporary_file_path = usage_file_path.with_extension("json.partial");
    let write_outcome = (|| {
        let serialized_usage = serde_json::to_vec(&usage_file).map_err(std::io::Error::other)?;
        std::fs::write(&temporary_file_path, serialized_usage)?;
        std::fs::rename(&temporary_file_path, usage_file_path)
    })();
    if let Err(write_error) = write_outcome {
        tracing::warn!(
            usage_file_path = usage_file_path.display().to_string(),
            error = %write_error,
            "expert usage history was not persisted; serving continues without it"
        );
        return false;
    }
    true
}

/// Reads the demand counts for a model with exactly `layer_count` layers.
/// Returns `None` for a missing, corrupt, foreign-format, or shape-mismatched
/// file — a stale history must never seed a differently shaped model.
pub(crate) fn read_expert_usage(
    usage_file_path: &Path,
    layer_count: usize,
) -> Option<Vec<Vec<u64>>> {
    if layer_count == 0 {
        return None;
    }
    let serialized_usage = std::fs::read(usage_file_path).ok()?;
    let usage_file: ExpertUsageFile = serde_json::from_slice(&serialized_usage).ok()?;
    if usage_file.format_version != EXPERT_USAGE_FORMAT_VERSION
        || usage_file.layer_count != layer_count
        || usage_file.layers.len() != layer_count
    {
        return None;
    }
    let mut demand_counts_by_layer = vec![Vec::new(); layer_count];
    let mut expert_capacity = 0_usize;
    for layer in &usage_file.layers {
        if layer.layer_index >= layer_count {
            return None;
        }
        expert_capacity = expert_capacity.max(
            layer
                .demand_counts_by_expert_id
                .keys()
                .next_back()
                .map_or(0, |largest_expert_id| largest_expert_id + 1),
        );
        demand_counts_by_layer[layer.layer_index] = layer
            .demand_counts_by_expert_id
            .iter()
            .map(|(expert_id, demand_count)| (*expert_id, *demand_count))
            .collect::<Vec<_>>();
    }
    // Densify into the [layer][expert] layout the cache records demand in.
    Some(
        demand_counts_by_layer
            .into_iter()
            .map(|layer_demand_counts| {
                let mut dense_counts = vec![0_u64; expert_capacity];
                for (expert_id, demand_count) in layer_demand_counts {
                    dense_counts[expert_id] = demand_count;
                }
                dense_counts
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "expert_usage_history_tests.rs"]
mod expert_usage_history_tests;

/// One layer's persisted expert-demand counts: the non-zero experts in
/// expert-id order, capped at the per-layer persistence limit so one hot
/// layer cannot crowd the rest out of the history file.
fn persisted_expert_usage_layer(
    layer_index: usize,
    layer_demand_counts: &[u64],
) -> ExpertUsageLayer {
    let mut demand_counts_by_expert_id = layer_demand_counts
        .iter()
        .enumerate()
        .filter(|(_, demand_count)| **demand_count > 0)
        .take(MAXIMUM_PERSISTED_EXPERTS_PER_LAYER)
        .map(|(expert_id, demand_count)| (expert_id, *demand_count))
        .collect::<BTreeMap<_, _>>();
    if demand_counts_by_expert_id.len() > MAXIMUM_PERSISTED_EXPERTS_PER_LAYER {
        demand_counts_by_expert_id = demand_counts_by_expert_id
            .into_iter()
            .take(MAXIMUM_PERSISTED_EXPERTS_PER_LAYER)
            .collect();
    }
    ExpertUsageLayer {
        layer_index,
        demand_counts_by_expert_id,
    }
}
