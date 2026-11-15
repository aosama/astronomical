//! Settings whose key changed: the old key's value moves to the new key.
//!
//! The configuration parses strictly, so a file written before a key was
//! renamed would stop loading after an update. Each rename here moves the
//! value before parsing, and the file is written back under the new key.

use serde_json::{Map, Value};

/// A root section whose key changed.
const RENAMED_ROOT_FIELDS: [(&str, &str); 1] = [
    // The cache of prompt blocks on the SSD is the `astro_ssd_prompt_cache`;
    // "prompt cache" alone did not say which cache.
    ("prompt_cache", "ssd_prompt_cache"),
];

/// A chunking setting whose key changed, in the global section and in each
/// model's.
const RENAMED_CHUNKING_FIELDS: [(&str, &str); 2] = [
    ("prompt_cache_block_tokens", "ssd_prompt_cache_block_tokens"),
    (
        "prompt_cache_common_prefix_stride_blocks",
        "ssd_prompt_cache_common_prefix_stride_blocks",
    ),
];

/// Moves every renamed setting to its new key. Returns whether anything
/// moved, so the caller writes the file back under the new keys.
///
/// When a file holds both keys, the new key's value wins and the old key is
/// dropped: the new key can only have been written by a newer Astronomical.
pub(crate) fn rename_renamed_config_fields(config_json: &mut Value) -> bool {
    let mut renamed_field_names = Vec::new();
    if let Some(root_object) = config_json.as_object_mut() {
        rename_fields(
            root_object,
            &RENAMED_ROOT_FIELDS,
            "",
            &mut renamed_field_names,
        );
        rename_chunking_fields(root_object, "chunking", &mut renamed_field_names);
        if let Some(models) = root_object.get_mut("models").and_then(Value::as_object_mut) {
            for (model_id, model_config) in models.iter_mut() {
                if let Some(model_object) = model_config.as_object_mut() {
                    let chunking_path = format!("models.{model_id}.chunking");
                    rename_chunking_fields(model_object, &chunking_path, &mut renamed_field_names);
                }
            }
        }
    }
    if !renamed_field_names.is_empty() {
        tracing::info!(
            renamed_fields = ?renamed_field_names,
            "moved configuration settings to their new keys; the configuration is persisted"
        );
    }
    !renamed_field_names.is_empty()
}

fn rename_chunking_fields(
    parent_object: &mut Map<String, Value>,
    chunking_path: &str,
    renamed_field_names: &mut Vec<String>,
) {
    if let Some(chunking_object) = parent_object
        .get_mut("chunking")
        .and_then(Value::as_object_mut)
    {
        let field_path_prefix = format!("{chunking_path}.");
        rename_fields(
            chunking_object,
            &RENAMED_CHUNKING_FIELDS,
            &field_path_prefix,
            renamed_field_names,
        );
    }
}

fn rename_fields(
    object: &mut Map<String, Value>,
    renamed_fields: &[(&str, &str)],
    field_path_prefix: &str,
    renamed_field_names: &mut Vec<String>,
) {
    for (old_field_name, new_field_name) in renamed_fields {
        let Some(old_field_value) = object.remove(*old_field_name) else {
            continue;
        };
        if !object.contains_key(*new_field_name) {
            object.insert((*new_field_name).to_owned(), old_field_value);
        }
        renamed_field_names.push(format!("{field_path_prefix}{old_field_name}"));
    }
}

#[cfg(test)]
#[path = "config_field_renames_tests.rs"]
mod config_field_renames_tests;
