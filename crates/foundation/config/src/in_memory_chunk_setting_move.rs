//! Moves the in-memory prompt chunk from its old key to its new one, once.
//!
//! Earlier builds wrote their own default chunk (2,048, and 4,096 before
//! that) into every `config.json` under `chunking.fixed_prompt_processing_chunk_size_tokens`.
//! A file holding that number cannot say whether a person chose it, so the
//! setting moved to `prompt_processing_chunk_size_tokens`, where `null` is
//! Automatic and the engine picks the size for the model and the Mac. Moving
//! the key is what makes the change happen exactly once: after it, a 2,048
//! under the new key can only be a person's choice and is kept.
//!
//! In the global section a number an earlier build wrote becomes `null`;
//! any other number moves unchanged. A model's own section was never
//! written by Astronomical, so its number always moves unchanged.

use serde_json::{Map, Value};

const OLD_IN_MEMORY_CHUNK_KEY: &str = "fixed_prompt_processing_chunk_size_tokens";
const NEW_IN_MEMORY_CHUNK_KEY: &str = "prompt_processing_chunk_size_tokens";

/// The chunk sizes earlier macOS builds wrote as their own default.
#[cfg(target_os = "macos")]
const CHUNK_SIZES_EARLIER_BUILDS_WROTE: [u64; 2] = [2_048, 4_096];
/// Builds without Metal wrote 64, and before that the macOS sizes.
#[cfg(not(target_os = "macos"))]
const CHUNK_SIZES_EARLIER_BUILDS_WROTE: [u64; 3] = [64, 2_048, 4_096];

/// Moves the global section's and every model's in-memory chunk to the new
/// key. Returns the moved settings' paths, so the caller logs them and writes
/// the file back.
pub(crate) fn move_in_memory_chunk_setting(root_object: &mut Map<String, Value>) -> Vec<String> {
    let mut moved_setting_paths = Vec::new();
    if let Some(global_chunking) = chunking_section(root_object)
        && move_chunk_key(global_chunking, true)
    {
        moved_setting_paths.push(format!("chunking.{OLD_IN_MEMORY_CHUNK_KEY}"));
    }
    if let Some(models) = root_object.get_mut("models").and_then(Value::as_object_mut) {
        for (model_id, model_config) in models.iter_mut() {
            let Some(model_chunking) = model_config.as_object_mut().and_then(chunking_section)
            else {
                continue;
            };
            if move_chunk_key(model_chunking, false) {
                moved_setting_paths.push(format!(
                    "models.{model_id}.chunking.{OLD_IN_MEMORY_CHUNK_KEY}"
                ));
            }
        }
    }
    moved_setting_paths
}

fn chunking_section(parent_object: &mut Map<String, Value>) -> Option<&mut Map<String, Value>> {
    parent_object
        .get_mut("chunking")
        .and_then(Value::as_object_mut)
}

/// Moves one section's chunk. When a file holds both keys the new one wins,
/// because only a newer Astronomical can have written it.
fn move_chunk_key(chunking: &mut Map<String, Value>, is_global_section: bool) -> bool {
    let Some(old_chunk_value) = chunking.remove(OLD_IN_MEMORY_CHUNK_KEY) else {
        return false;
    };
    if !chunking.contains_key(NEW_IN_MEMORY_CHUNK_KEY) {
        let new_chunk_value =
            if is_global_section && was_written_by_an_earlier_build(&old_chunk_value) {
                Value::Null
            } else {
                old_chunk_value
            };
        chunking.insert(NEW_IN_MEMORY_CHUNK_KEY.to_owned(), new_chunk_value);
    }
    true
}

fn was_written_by_an_earlier_build(chunk_value: &Value) -> bool {
    match chunk_value.as_u64() {
        Some(chunk_tokens) => CHUNK_SIZES_EARLIER_BUILDS_WROTE.contains(&chunk_tokens),
        None => false,
    }
}

#[cfg(test)]
#[path = "in_memory_chunk_setting_move_tests.rs"]
mod in_memory_chunk_setting_move_tests;
