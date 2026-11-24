//! The JSON documents of a model with an added tower. Pure, so each rule can be
//! tested without any files.

use serde_json::{Map, Value, json};

const TOWER_TOKEN_FIELDS: [&str; 4] = [
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
];

/// Where the tower is written. A survey of 436 published MLX models with towers
/// found 94 percent keep them in `model*` shards, and this name is the most common
/// choice among those that give the tower its own file. Loaders find it through the
/// index: mlx-vlm and Astronomical read the index, and mlx-lm's Qwen3.5 loader drops
/// `vision_tower` tensors, so it still loads text only. Other mlx-lm families have
/// not been checked. OptiQ's engine looks only for its own `optiq_vision.safetensors`
/// sidecar, so it reads this model as text only. That name is not used here because
/// it is OptiQ's packaging convention, and these artifacts are not OptiQ builds.
pub const TOWER_FILE_RELATIVE_PATH: &str = "model-vision.safetensors";

/// Facts about the tower file that the documents record.
#[derive(Clone, Copy, Debug)]
pub struct TowerDocumentFacts<'facts> {
    pub tower_config: &'facts Value,
    pub tensor_names: &'facts [String],
    pub tensor_byte_count: u64,
    pub parameter_count: u64,
}

/// The text model's config plus the tower's section. The text settings are left
/// exactly as published, because the tower must not change how text runs.
#[must_use]
pub fn config_with_tower(text_config: &Value, facts: &TowerDocumentFacts<'_>) -> Value {
    let mut merged_config = match text_config {
        Value::Object(config_fields) => config_fields.clone(),
        _ => Map::new(),
    };
    if let Some(vision_config) = facts.tower_config.get("vision_config") {
        merged_config.insert("vision_config".to_owned(), vision_config.clone());
    }
    for token_field in TOWER_TOKEN_FIELDS {
        if !merged_config.contains_key(token_field)
            && let Some(tower_token) = facts.tower_config.get(token_field)
        {
            merged_config.insert(token_field.to_owned(), tower_token.clone());
        }
    }
    Value::Object(merged_config)
}

/// The text model's index with the tower added. Loaders and Astronomical's discovery
/// go by this file, so a tower missing from it is a tower nobody finds.
#[must_use]
pub fn index_with_tower(
    text_index: Option<&Value>,
    text_tensor_files: &[(String, String)],
    facts: &TowerDocumentFacts<'_>,
) -> Value {
    let mut weight_map = Map::new();
    let existing_weight_map = match text_index {
        Some(text_index) => text_index.get("weight_map").and_then(Value::as_object),
        None => None,
    };
    if let Some(existing_weight_map) = existing_weight_map {
        weight_map = existing_weight_map.clone();
    } else {
        for (tensor_name, relative_path) in text_tensor_files {
            weight_map.insert(tensor_name.clone(), Value::String(relative_path.clone()));
        }
    }
    for tensor_name in facts.tensor_names {
        weight_map.insert(
            tensor_name.clone(),
            Value::String(TOWER_FILE_RELATIVE_PATH.to_owned()),
        );
    }
    let mut metadata = match text_index.and_then(metadata_of) {
        Some(metadata) => metadata.clone(),
        None => Map::new(),
    };
    add_to_field(&mut metadata, "total_size", facts.tensor_byte_count, true);
    add_to_field(
        &mut metadata,
        "total_parameters",
        facts.parameter_count,
        false,
    );
    json!({"metadata": metadata, "weight_map": weight_map})
}

fn metadata_of(text_index: &Value) -> Option<&Map<String, Value>> {
    text_index.get("metadata")?.as_object()
}

/// Adds to a running total. A total the source never stated stays unstated unless
/// it is one the format expects, since a made-up partial count would mislead.
fn add_to_field(
    metadata: &mut Map<String, Value>,
    field_name: &str,
    added_count: u64,
    is_created_when_missing: bool,
) {
    let existing_count = metadata.get(field_name).and_then(Value::as_u64);
    match existing_count {
        Some(existing_count) => {
            metadata.insert(field_name.to_owned(), json!(existing_count + added_count));
        }
        None if is_created_when_missing => {
            metadata.insert(field_name.to_owned(), json!(added_count));
        }
        None => {}
    }
}

#[cfg(test)]
#[path = "assembled_documents_tests.rs"]
mod tests;
