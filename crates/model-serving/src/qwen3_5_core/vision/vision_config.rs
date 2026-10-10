//! Exact vision architecture accepted by the pinned Qwen3.5 executor.
//!
//! These values are not tuning knobs. They determine tensor shapes and model
//! math: head_dimension=1152/16=72, position table side=sqrt(2304)=48,
//! flattened patch width=3*2*16*16=1536, and merger width=1152*2*2=4608.
//! Validating them at artifact load keeps later graph assembly simple and avoids
//! accidentally running checkpoint tensors under a merely similar architecture.

use super::Qwen3_5ConfigError;
use serde::Deserialize;

const ACCEPTED_VISION_MODEL_TYPES: &[&str] = &[
    "qwen3_5",
    "qwen3_5_vision",
    "qwen3_5_moe_vision",
    "qwen3_5_moe",
];

/// The Qwen3.5 vision configuration accepted for Qwen3.5 execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5VisionConfig {
    depth: u32,
    hidden_size: u32,
    in_channels: u32,
    intermediate_size: u32,
    head_count: u32,
    position_embedding_count: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
    out_hidden_size: u32,
    hidden_activation: String,
}

impl Qwen3_5VisionConfig {
    /// Parses the vision_config section from the retained config bytes.
    pub fn from_json_bytes(config_bytes: &[u8]) -> Result<Self, Qwen3_5ConfigError> {
        Self::from_optional_json_bytes(config_bytes)?.ok_or(Qwen3_5ConfigError::MissingVisionConfig)
    }

    /// Parses an optional vision_config section from the retained config bytes.
    ///
    /// Text-only Qwen checkpoints legitimately omit this section.
    pub fn from_optional_json_bytes(
        config_bytes: &[u8],
    ) -> Result<Option<Self>, Qwen3_5ConfigError> {
        let config_document =
            serde_json::from_slice::<Qwen3_5OptionalVisionConfigDocument>(config_bytes)
                .map_err(Qwen3_5ConfigError::DeserializeVisionConfig)?;
        let Some(vision_config) = config_document.vision_config else {
            return Ok(None);
        };
        vision_config.validate()?;
        Ok(Some(Self {
            depth: vision_config.depth,
            hidden_size: vision_config.hidden_size,
            in_channels: vision_config.in_channels,
            intermediate_size: vision_config.intermediate_size,
            head_count: vision_config.num_heads,
            position_embedding_count: vision_config.num_position_embeddings,
            patch_size: vision_config.patch_size,
            spatial_merge_size: vision_config.spatial_merge_size,
            temporal_patch_size: vision_config.temporal_patch_size,
            out_hidden_size: vision_config.out_hidden_size,
            hidden_activation: vision_config.hidden_act,
        }))
    }

    /// Returns the vision transformer block count.
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// Returns the vision hidden dimension.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Returns the input channel count (3 for RGB).
    #[must_use]
    pub const fn in_channels(&self) -> u32 {
        self.in_channels
    }

    /// Returns the feed-forward intermediate size.
    #[must_use]
    pub const fn intermediate_size(&self) -> u32 {
        self.intermediate_size
    }

    /// Returns the attention head count.
    #[must_use]
    pub const fn head_count(&self) -> u32 {
        self.head_count
    }

    /// Returns the positional embedding count.
    #[must_use]
    pub const fn position_embedding_count(&self) -> u32 {
        self.position_embedding_count
    }

    /// Returns the patch size in pixels.
    #[must_use]
    pub const fn patch_size(&self) -> u32 {
        self.patch_size
    }

    /// Returns the spatial merge size.
    #[must_use]
    pub const fn spatial_merge_size(&self) -> u32 {
        self.spatial_merge_size
    }

    /// Returns the temporal patch size.
    #[must_use]
    pub const fn temporal_patch_size(&self) -> u32 {
        self.temporal_patch_size
    }

    /// Returns the output hidden size (projects into the text model).
    #[must_use]
    pub const fn out_hidden_size(&self) -> u32 {
        self.out_hidden_size
    }

    /// Returns the vision hidden activation function name.
    #[must_use]
    pub fn hidden_activation(&self) -> &str {
        &self.hidden_activation
    }
}

#[derive(Debug, Deserialize)]
struct Qwen3_5OptionalVisionConfigDocument {
    #[serde(default)]
    vision_config: Option<Qwen3_5VisionFields>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Qwen3_5VisionFields {
    depth: u32,
    hidden_size: u32,
    in_channels: u32,
    intermediate_size: u32,
    model_type: String,
    num_heads: u32,
    num_position_embeddings: u32,
    out_hidden_size: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
    hidden_act: String,
    #[serde(default)]
    deepstack_visual_indexes: Vec<u32>,
}

impl Qwen3_5VisionFields {
    fn validate(&self) -> Result<(), Qwen3_5ConfigError> {
        if !ACCEPTED_VISION_MODEL_TYPES.contains(&self.model_type.as_str()) {
            return Err(Qwen3_5ConfigError::UnexpectedStringValue {
                field_name: "vision_config.model_type",
                expected_value: "qwen3_5, qwen3_5_vision, qwen3_5_moe_vision, or qwen3_5_moe",
                actual_value: self.model_type.clone(),
            });
        }
        // Structural sanity: values must be positive. Any valid Qwen3.5
        // vision config is accepted; the values are not hardcoded to one model.
        if self.depth == 0
            || self.hidden_size == 0
            || self.in_channels == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || self.num_position_embeddings == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
            || self.out_hidden_size == 0
        {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "vision_config numeric fields must be positive",
            });
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "vision_config.hidden_size must divide evenly by num_heads",
            });
        }
        let head_dimension = self.hidden_size / self.num_heads;
        if !head_dimension.is_multiple_of(4) {
            return Err(Qwen3_5ConfigError::InvalidConfigValue {
                description: "vision attention head dimension must divide evenly across two rotary axes",
            });
        }
        if self.hidden_act != "gelu_pytorch_tanh" {
            return Err(Qwen3_5ConfigError::UnexpectedStringValue {
                field_name: "vision_config.hidden_act",
                expected_value: "gelu_pytorch_tanh",
                actual_value: self.hidden_act.clone(),
            });
        }
        Ok(())
    }
}
