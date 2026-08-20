//! Response ownership for one OpenAI-compatible base64 image result.

use serde::Serialize;

/// Validated generated-image content and reproducibility metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiGeneratedImageParts {
    pub b64_json: String,
    pub mime_type: String,
    pub model_revision: String,
    pub effective_seed: u64,
    pub width: u32,
    pub height: u32,
}

/// One OpenAI-compatible image generation response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiImageGenerationResponse {
    created: u64,
    data: Vec<OpenAiGeneratedImage>,
}

impl OpenAiImageGenerationResponse {
    /// Builds the single-image response supported by the initial endpoint.
    #[must_use]
    pub fn new(created: u64, generated_image_parts: OpenAiGeneratedImageParts) -> Self {
        Self {
            created,
            data: vec![OpenAiGeneratedImage {
                b64_json: generated_image_parts.b64_json,
                mime_type: generated_image_parts.mime_type,
                model_revision: generated_image_parts.model_revision,
                effective_seed: generated_image_parts.effective_seed,
                width: generated_image_parts.width,
                height: generated_image_parts.height,
            }],
        }
    }
}

/// One encoded image returned inside an image generation response.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct OpenAiGeneratedImage {
    b64_json: String,
    mime_type: String,
    model_revision: String,
    #[serde(rename = "seed")]
    effective_seed: u64,
    width: u32,
    height: u32,
}
