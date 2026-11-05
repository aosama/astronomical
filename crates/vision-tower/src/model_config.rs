//! The few `config.json` facts a tower has to agree with. Fields are optional
//! because configs differ across families and a missing one is itself a finding.

use serde::Serialize;
use serde_json::Value;

use crate::artifact_source::{ArtifactSource, CONFIG_FILE_NAME};
use crate::error::VisionTowerError;

const MAXIMUM_CONFIG_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ModelConfigSummary {
    pub model_type: Option<String>,
    pub text_hidden_size: Option<u64>,
    pub image_token_id: Option<u64>,
    pub video_token_id: Option<u64>,
    pub vision_start_token_id: Option<u64>,
    pub vision_end_token_id: Option<u64>,
    pub vision: Option<VisionConfigSummary>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct VisionConfigSummary {
    pub model_type: Option<String>,
    pub block_count: Option<u64>,
    pub hidden_size: Option<u64>,
    pub output_hidden_size: Option<u64>,
    pub patch_size: Option<u64>,
    pub spatial_merge_size: Option<u64>,
}

impl ModelConfigSummary {
    #[must_use]
    pub const fn declares_image_tokens(&self) -> bool {
        self.image_token_id.is_some()
    }

    #[must_use]
    pub fn from_config_value(config_value: &Value) -> Self {
        let text_config = config_value.get("text_config");
        // Multimodal configs nest the text width; plain ones state it at the root.
        let text_hidden_size = match text_config.and_then(hidden_size_of) {
            Some(nested_hidden_size) => Some(nested_hidden_size),
            None => hidden_size_of(config_value),
        };
        Self {
            model_type: text_field(config_value, "model_type"),
            text_hidden_size,
            image_token_id: unsigned_field(config_value, "image_token_id"),
            video_token_id: unsigned_field(config_value, "video_token_id"),
            vision_start_token_id: unsigned_field(config_value, "vision_start_token_id"),
            vision_end_token_id: unsigned_field(config_value, "vision_end_token_id"),
            vision: vision_summary(config_value.get("vision_config")),
        }
    }
}

pub async fn read_model_config(
    source: &ArtifactSource,
) -> Result<ModelConfigSummary, VisionTowerError> {
    let Some(config_bytes) = source
        .read_whole_file(CONFIG_FILE_NAME, MAXIMUM_CONFIG_BYTES)
        .await?
    else {
        return Ok(ModelConfigSummary::default());
    };
    match serde_json::from_slice::<Value>(&config_bytes) {
        Ok(config_value) => Ok(ModelConfigSummary::from_config_value(&config_value)),
        Err(parse_error) => Err(VisionTowerError::JsonMalformed {
            file: CONFIG_FILE_NAME.to_owned(),
            description: parse_error.to_string(),
        }),
    }
}

fn vision_summary(vision_config: Option<&Value>) -> Option<VisionConfigSummary> {
    let vision_config = vision_config?;
    if !vision_config.is_object() {
        return None;
    }
    Some(VisionConfigSummary {
        model_type: text_field(vision_config, "model_type"),
        block_count: unsigned_field(vision_config, "depth"),
        hidden_size: hidden_size_of(vision_config),
        output_hidden_size: unsigned_field(vision_config, "out_hidden_size"),
        patch_size: unsigned_field(vision_config, "patch_size"),
        spatial_merge_size: unsigned_field(vision_config, "spatial_merge_size"),
    })
}

fn hidden_size_of(config_section: &Value) -> Option<u64> {
    unsigned_field(config_section, "hidden_size")
}

fn unsigned_field(config_section: &Value, field_name: &str) -> Option<u64> {
    config_section.get(field_name).and_then(Value::as_u64)
}

fn text_field(config_section: &Value, field_name: &str) -> Option<String> {
    config_section
        .get(field_name)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

#[cfg(test)]
#[path = "model_config_tests.rs"]
mod tests;
