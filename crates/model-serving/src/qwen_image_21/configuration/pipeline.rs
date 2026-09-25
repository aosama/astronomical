//! `model_index.json`: the pipeline ownership graph.

use serde::Deserialize;

use super::{QwenImage21ConfigError, parse_document, require};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    processor: [String; 2],
    scheduler: [String; 2],
    text_encoder: [String; 2],
    transformer: [String; 2],
    vae: [String; 2],
}

/// Validated ownership graph from `model_index.json`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21PipelineConfig;

impl QwenImage21PipelineConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, QwenImage21ConfigError> {
        const DOCUMENT: &str = "model_index.json";
        let document: PipelineDocument = parse_document(json_bytes, DOCUMENT)?;
        require(
            document.class_name == "QwenImage21Pipeline",
            DOCUMENT,
            "_class_name",
        )?;
        require(
            document.diffusers_version == "0.37.0.dev0",
            DOCUMENT,
            "_diffusers_version",
        )?;
        require(
            document.processor == ["transformers", "Qwen3VLProcessor"],
            DOCUMENT,
            "processor",
        )?;
        require(
            document.scheduler == ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            DOCUMENT,
            "scheduler",
        )?;
        require(
            document.text_encoder == ["transformers", "Qwen3VLForConditionalGeneration"],
            DOCUMENT,
            "text_encoder",
        )?;
        require(
            document.transformer == ["diffusers", "QwenImage21Transformer2DModel"],
            DOCUMENT,
            "transformer",
        )?;
        require(
            document.vae == ["diffusers", "AutoencoderKLQwenImage21"],
            DOCUMENT,
            "vae",
        )?;
        Ok(Self)
    }
}
