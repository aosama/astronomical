//! Resumable sweep state on disk.
//!
//! A sweep on a model that streams from SSD runs for hours, so a killed run
//! must resume. The checkpoint is a JSON array of serialized
//! [`SensitivityResult`]s in OptiQ's `sensitivity_checkpoint.json` shape
//! (`layer_name`, `sensitivities` keyed by the bit width as a string,
//! `param_count`), so a checkpoint written by the Python tool resumes here and
//! vice versa.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bit_width::BitWidth;
use crate::error::QuantizationError;

use super::SensitivityResult;

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod checkpoint_tests;

/// Loads the checkpoint, ignoring an unreadable file: a corrupt checkpoint
/// costs a re-sweep, never a refusal to run.
///
/// Each entry's `layer_name`, string-keyed `sensitivities`, and `param_count`
/// translate to a [`SensitivityResult`]. Python writes a non-finite KL as a
/// bare `NaN` literal, which is not JSON, so bare non-finite literals are
/// quoted before parsing and read back as non-finite values.
#[must_use]
pub fn load_checkpoint(checkpoint_path: &Path) -> Vec<SensitivityResult> {
    tracing::debug!(checkpoint_path = %checkpoint_path.display(), "loading the sensitivity checkpoint");
    let Some(checkpoint_entries) = read_checkpoint_entries(checkpoint_path) else {
        return Vec::new();
    };
    let mut results = Vec::with_capacity(checkpoint_entries.len());
    for checkpoint_entry in &checkpoint_entries {
        match result_from_entry(checkpoint_entry) {
            Some(result) => results.push(result),
            None => {
                tracing::warn!(
                    checkpoint_path = %checkpoint_path.display(),
                    layer_name = %checkpoint_entry.layer_name,
                    "ignoring the checkpoint: an entry has an unsupported bit width"
                );
                return Vec::new();
            }
        }
    }
    results
}

/// Persists results by merging with the existing file, so it only grows.
///
/// A resumed run re-appends already-complete modules to its in-memory
/// results only as the loop passes them, so a periodic save (or a kill)
/// mid-loop could otherwise write fewer entries than the file held and drop
/// scored modules. Entries merge by module name (existing, then current
/// results replacing or appending), and the file is written to `<path>.tmp`
/// and renamed over the original so a crash mid-save cannot corrupt it.
pub fn merge_and_save_checkpoint(
    checkpoint_path: &Path,
    results: &[SensitivityResult],
) -> Result<(), QuantizationError> {
    let mut merged_entries = read_checkpoint_entries(checkpoint_path).unwrap_or_default();
    for result in results {
        let current_entry = entry_from_result(result);
        let mut replaced_existing = false;
        for existing_entry in &mut merged_entries {
            if existing_entry.layer_name == current_entry.layer_name {
                *existing_entry = current_entry.clone();
                replaced_existing = true;
            }
        }
        if !replaced_existing {
            merged_entries.push(current_entry);
        }
    }
    write_entries_atomically(checkpoint_path, &merged_entries)
}

/// Keeps only the checkpointed entries that cover every requested width;
/// an entry from a sweep with a different ladder must be re-probed.
#[must_use]
pub fn complete_entries_for(
    checkpointed: Vec<SensitivityResult>,
    candidate_bits: &[BitWidth],
) -> Vec<SensitivityResult> {
    let mut complete_entries = Vec::with_capacity(checkpointed.len());
    for checkpointed_result in checkpointed {
        let mut covers_every_width = true;
        for bit_width in candidate_bits {
            if !checkpointed_result
                .kl_divergence_by_bit_width
                .contains_key(bit_width)
            {
                covers_every_width = false;
            }
        }
        if covers_every_width {
            complete_entries.push(checkpointed_result);
        }
    }
    complete_entries
}

/// One entry in OptiQ's checkpoint shape. `metadata` is carried through
/// untouched so a Python checkpoint keeps its per-layer notes.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointEntry {
    layer_name: String,
    sensitivities: BTreeMap<String, serde_json::Value>,
    param_count: u64,
    #[serde(default)]
    metadata: serde_json::Value,
}

/// The entries on disk, or `None` when the file is absent or malformed.
fn read_checkpoint_entries(checkpoint_path: &Path) -> Option<Vec<CheckpointEntry>> {
    let checkpoint_text = std::fs::read_to_string(checkpoint_path).ok()?;
    match serde_json::from_str(&quote_bare_non_finite_literals(&checkpoint_text)) {
        Ok(checkpoint_entries) => Some(checkpoint_entries),
        Err(parse_error) => {
            tracing::warn!(
                checkpoint_path = %checkpoint_path.display(),
                %parse_error,
                "ignoring a malformed sensitivity checkpoint"
            );
            None
        }
    }
}

fn result_from_entry(checkpoint_entry: &CheckpointEntry) -> Option<SensitivityResult> {
    let mut kl_divergence_by_bit_width = BTreeMap::new();
    for (bits_key, kl_value) in &checkpoint_entry.sensitivities {
        let bit_width = BitWidth::new(bits_key.parse().ok()?).ok()?;
        kl_divergence_by_bit_width.insert(bit_width, kl_divergence_of(kl_value)?);
    }
    Some(SensitivityResult {
        module_name: checkpoint_entry.layer_name.clone(),
        kl_divergence_by_bit_width,
        logical_parameter_count: checkpoint_entry.param_count,
    })
}

/// A KL value as a number, or a quoted non-finite literal.
fn kl_divergence_of(kl_value: &serde_json::Value) -> Option<f64> {
    if let Some(kl_divergence) = kl_value.as_f64() {
        return Some(kl_divergence);
    }
    match kl_value.as_str()? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// Non-finite values are written as the strings Python's `float()` accepts,
/// since JSON has no literal for them.
fn entry_from_result(result: &SensitivityResult) -> CheckpointEntry {
    let mut sensitivities = BTreeMap::new();
    for (bit_width, kl_divergence) in &result.kl_divergence_by_bit_width {
        let kl_value = if kl_divergence.is_nan() {
            serde_json::Value::from("NaN")
        } else if kl_divergence.is_infinite() {
            let infinity_text = if *kl_divergence > 0.0 {
                "Infinity"
            } else {
                "-Infinity"
            };
            serde_json::Value::from(infinity_text)
        } else {
            serde_json::Value::from(*kl_divergence)
        };
        sensitivities.insert(bit_width.bits().to_string(), kl_value);
    }
    CheckpointEntry {
        layer_name: result.module_name.clone(),
        sensitivities,
        param_count: result.logical_parameter_count,
        metadata: serde_json::Value::Object(serde_json::Map::new()),
    }
}

fn write_entries_atomically(
    checkpoint_path: &Path,
    checkpoint_entries: &[CheckpointEntry],
) -> Result<(), QuantizationError> {
    let mut temporary_path = checkpoint_path.as_os_str().to_owned();
    temporary_path.push(".tmp");
    let temporary_path = PathBuf::from(temporary_path);
    let checkpoint_text = match serde_json::to_string_pretty(checkpoint_entries) {
        Ok(checkpoint_text) => checkpoint_text,
        Err(serialize_error) => {
            return Err(checkpoint_unwritable(
                checkpoint_path,
                &serialize_error.to_string(),
            ));
        }
    };
    if let Err(write_error) = std::fs::write(&temporary_path, checkpoint_text) {
        return Err(checkpoint_unwritable(
            checkpoint_path,
            &write_error.to_string(),
        ));
    }
    match std::fs::rename(&temporary_path, checkpoint_path) {
        Ok(()) => Ok(()),
        Err(rename_error) => Err(checkpoint_unwritable(
            checkpoint_path,
            &rename_error.to_string(),
        )),
    }
}

fn checkpoint_unwritable(checkpoint_path: &Path, description: &str) -> QuantizationError {
    QuantizationError::CheckpointUnwritable {
        checkpoint_path: checkpoint_path.to_path_buf(),
        description: description.to_owned(),
    }
}

/// Quotes Python's bare `NaN`, `Infinity`, and `-Infinity` literals that
/// appear outside strings, leaving string contents untouched.
fn quote_bare_non_finite_literals(json_text: &str) -> String {
    let mut quoted_text = String::with_capacity(json_text.len());
    let mut is_inside_string = false;
    let mut is_escaped = false;
    let mut remaining_text = json_text;
    while let Some(next_character) = remaining_text.chars().next() {
        if !is_inside_string
            && let Some((literal, literal_length)) = bare_literal_at(remaining_text)
        {
            quoted_text.push('"');
            quoted_text.push_str(literal);
            quoted_text.push('"');
            remaining_text = &remaining_text[literal_length..];
            continue;
        }
        if is_inside_string {
            is_inside_string = is_escaped || next_character != '"';
            is_escaped = !is_escaped && next_character == '\\';
        } else if next_character == '"' {
            is_inside_string = true;
        }
        quoted_text.push(next_character);
        remaining_text = &remaining_text[next_character.len_utf8()..];
    }
    quoted_text
}

/// The non-finite literal starting `text`, with its length.
fn bare_literal_at(text: &str) -> Option<(&'static str, usize)> {
    for literal in ["-Infinity", "Infinity", "NaN"] {
        if text.starts_with(literal) {
            return Some((literal, literal.len()));
        }
    }
    None
}
