//! Persistent per-model expert routing heat, split by serving phase
//! (colibri's `.coli_usage` analog).
//!
//! Decode's first touch of an expert is always a storage miss, so a session
//! starts cold no matter how large the warm tables are. This store persists
//! the per-layer expert demand counts the cache already records during
//! forwards — kept as two disjoint records, one for prompt prefill evidence
//! and one for decode evidence — beside the model artifact, and reloads them
//! at engine construction so historically hot experts are admitted warm and
//! ranked ahead of noise from the first token.
//!
//! The phase split exists because the two phases answer different questions:
//! prefill demand measures the breadth a prompt's route union needed (wide
//! and shallow — a long prompt touches many experts a handful of times),
//! while decode demand measures the depth a conversation actually keeps
//! re-using (narrow and hot). Fusing them into one counter — the format's
//! first version — let a long prompt's breadth masquerade as decode heat in
//! the next session's admissions, so version 2 keeps the records disjoint and
//! only the decode record seeds warm-table hotness.
//!
//! The file is derived from the runtime model directory, never a configured
//! or hardwired path. Every failure is non-fatal: an unreadable or stale file
//! leaves the session exactly as cold as it would have been without it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const EXPERT_USAGE_FILE_NAME: &str = ".astronomical-expert-usage.json";
/// Version 2 splits the per-layer counts into disjoint prefill and decode
/// records. Files from version 1 held the two phases fused, and fused counts
/// cannot be attributed back to a phase, so they are rejected rather than
/// mis-seeded: one cold start per model install is the honest cost.
const EXPERT_USAGE_FORMAT_VERSION: u32 = 2;
/// Per-layer cap on persisted entries: routing is sparse, and the cap keeps
/// the file bounded for artifacts with thousands of experts per layer. Each
/// phase is capped independently so a prompt-heavy generation cannot crowd
/// the decode record out of the history file.
const MAXIMUM_PERSISTED_EXPERTS_PER_LAYER: usize = 4_096;

/// The two phase-split demand records persisted between sessions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct PersistedExpertUsage {
    /// Prompt-prefill demand counts: what the route union of processed
    /// prompts needed, aged across generations.
    pub prefill_demand_counts_by_layer: Vec<Vec<u64>>,
    /// Decode demand counts: what generated tokens kept re-using, aged
    /// across generations. Only this record seeds warm-table hotness.
    pub decode_demand_counts_by_layer: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExpertUsageFile {
    format_version: u32,
    model_id: Option<String>,
    layer_count: usize,
    /// Sparse per-layer demand counts keyed by expert id, one map per phase.
    layers: Vec<ExpertUsageLayer>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExpertUsageLayer {
    layer_index: usize,
    prefill_demand_counts_by_expert_id: BTreeMap<usize, u64>,
    decode_demand_counts_by_expert_id: BTreeMap<usize, u64>,
}

/// Resolves the usage-history file inside a model directory.
#[must_use]
pub(crate) fn expert_usage_file_path(model_directory: &Path) -> PathBuf {
    model_directory.join(EXPERT_USAGE_FILE_NAME)
}

/// Atomically writes the phase-split demand records; returns whether the file
/// landed. A failed write logs and returns false — serving must never fail on
/// history.
pub(crate) fn write_expert_usage(
    usage_file_path: &Path,
    model_id: Option<&str>,
    persisted_usage: &PersistedExpertUsage,
) -> bool {
    let usage_file = ExpertUsageFile {
        format_version: EXPERT_USAGE_FORMAT_VERSION,
        model_id: model_id.map(str::to_owned),
        layer_count: persisted_usage.prefill_demand_counts_by_layer.len(),
        layers: persisted_usage_layers(persisted_usage),
    };
    let temporary_file_path = usage_file_path.with_extension("json.partial");
    let write_outcome = persist_usage_file_bytes(
        &temporary_file_path,
        usage_file_path,
        serde_json::to_vec(&usage_file).map_err(std::io::Error::other),
    );
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

/// Writes the serialized usage bytes to a temporary file, then renames it
/// over the destination so a crash can never leave a half-written history.
fn persist_usage_file_bytes(
    temporary_file_path: &Path,
    usage_file_path: &Path,
    serialized_usage: Result<Vec<u8>, std::io::Error>,
) -> Result<(), std::io::Error> {
    let serialized_usage = serialized_usage?;
    std::fs::write(temporary_file_path, serialized_usage)?;
    std::fs::rename(temporary_file_path, usage_file_path)
}

/// Builds the sparse per-layer file records from the dense phase-split
/// counters, applying the per-phase persistence cap per layer.
fn persisted_usage_layers(persisted_usage: &PersistedExpertUsage) -> Vec<ExpertUsageLayer> {
    let mut layers = Vec::with_capacity(persisted_usage.prefill_demand_counts_by_layer.len());
    for (layer_index, (prefill_layer_counts, decode_layer_counts)) in persisted_usage
        .prefill_demand_counts_by_layer
        .iter()
        .zip(persisted_usage.decode_demand_counts_by_layer.iter())
        .enumerate()
    {
        layers.push(ExpertUsageLayer {
            layer_index,
            prefill_demand_counts_by_expert_id: persisted_expert_usage_map(prefill_layer_counts),
            decode_demand_counts_by_expert_id: persisted_expert_usage_map(decode_layer_counts),
        });
    }
    layers
}

/// Reads the phase-split demand records for a model with exactly
/// `layer_count` layers. Returns `None` for a missing, corrupt, foreign
/// — including version-1 — or shape-mismatched file: a stale history must
/// never seed a differently shaped model.
pub(crate) fn read_expert_usage(
    usage_file_path: &Path,
    layer_count: usize,
) -> Option<PersistedExpertUsage> {
    if layer_count == 0 {
        return None;
    }
    let serialized_usage = std::fs::read(usage_file_path).ok()?;
    let usage_file: ExpertUsageFile = serde_json::from_slice(&serialized_usage).ok()?;
    if !usage_file_matches_layer_shape(&usage_file, layer_count) {
        return None;
    }
    Some(densified_persisted_usage(&usage_file, layer_count))
}

/// Validates the file header and layer indexing against the model's shape.
/// Version mismatches land here too: a version-1 file carries fused counts
/// that must not seed phase-split records.
fn usage_file_matches_layer_shape(usage_file: &ExpertUsageFile, layer_count: usize) -> bool {
    usage_file.format_version == EXPERT_USAGE_FORMAT_VERSION
        && usage_file.layer_count == layer_count
        && usage_file.layers.len() == layer_count
        && usage_file
            .layers
            .iter()
            .all(|layer| layer.layer_index < layer_count)
}

/// Densifies every layer's sparse maps into the `[layer][expert]` layout the
/// cache records demand in, with each phase densified to its own capacity.
fn densified_persisted_usage(
    usage_file: &ExpertUsageFile,
    layer_count: usize,
) -> PersistedExpertUsage {
    let mut prefill_capacity = 0_usize;
    let mut decode_capacity = 0_usize;
    for layer in &usage_file.layers {
        prefill_capacity = prefill_capacity.max(largest_expert_id_plus_one(
            &layer.prefill_demand_counts_by_expert_id,
        ));
        decode_capacity = decode_capacity.max(largest_expert_id_plus_one(
            &layer.decode_demand_counts_by_expert_id,
        ));
    }
    let mut prefill_demand_counts_by_layer = vec![Vec::new(); layer_count];
    let mut decode_demand_counts_by_layer = vec![Vec::new(); layer_count];
    for layer in &usage_file.layers {
        prefill_demand_counts_by_layer[layer.layer_index] =
            densified_demand_counts(&layer.prefill_demand_counts_by_expert_id, prefill_capacity);
        decode_demand_counts_by_layer[layer.layer_index] =
            densified_demand_counts(&layer.decode_demand_counts_by_expert_id, decode_capacity);
    }
    PersistedExpertUsage {
        prefill_demand_counts_by_layer,
        decode_demand_counts_by_layer,
    }
}

/// Largest expert id in a sparse map plus one, zero for an empty map: the
/// dense vector length the map densifies into.
fn largest_expert_id_plus_one(demand_counts_by_expert_id: &BTreeMap<usize, u64>) -> usize {
    match demand_counts_by_expert_id.keys().next_back() {
        Some(largest_expert_id) => largest_expert_id + 1,
        None => 0,
    }
}

/// Densifies one layer's sparse map into the `[expert]` layout the cache
/// records demand in.
fn densified_demand_counts(
    demand_counts_by_expert_id: &BTreeMap<usize, u64>,
    expert_capacity: usize,
) -> Vec<u64> {
    let mut dense_counts = vec![0_u64; expert_capacity];
    for (expert_id, demand_count) in demand_counts_by_expert_id {
        if *expert_id < expert_capacity {
            dense_counts[*expert_id] = *demand_count;
        }
    }
    dense_counts
}

/// One layer's non-zero demand counts in expert-id order for one phase,
/// capped at the per-layer persistence limit so one hot layer cannot crowd
/// the rest of the history file.
fn persisted_expert_usage_map(layer_demand_counts: &[u64]) -> BTreeMap<usize, u64> {
    let mut demand_counts_by_expert_id = BTreeMap::new();
    for (expert_id, demand_count) in layer_demand_counts.iter().enumerate() {
        if demand_counts_by_expert_id.len() >= MAXIMUM_PERSISTED_EXPERTS_PER_LAYER {
            break;
        }
        if *demand_count > 0 {
            demand_counts_by_expert_id.insert(expert_id, *demand_count);
        }
    }
    demand_counts_by_expert_id
}

#[cfg(test)]
#[path = "expert_usage_history_tests.rs"]
mod expert_usage_history_tests;
