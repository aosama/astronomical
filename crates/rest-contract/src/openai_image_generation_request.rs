//! Strict request validation for the OpenAI-compatible image-generation boundary.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// Smallest image side supported by the initial native image profile.
pub const MIN_OPENAI_IMAGE_DIMENSION_PIXELS: u32 = 64;
/// Largest image side supported by the initial native image profile.
pub const MAX_OPENAI_IMAGE_DIMENSION_PIXELS: u32 = 1_024;

const IMAGE_DIMENSION_MULTIPLE_PIXELS: u32 = 16;
const SUPPORTED_IMAGE_STEPS: u32 = 4;
const SUPPORTED_IMAGE_GUIDANCE: f32 = 1.0;

/// One strict request to the local OpenAI-compatible image generation endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct OpenAiImageGenerationRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    seed: Option<u64>,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    response_format: String,
    #[serde(default = "default_image_count")]
    n: u32,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

impl OpenAiImageGenerationRequest {
    /// Validates and consumes public input before queue admission or model loading.
    pub fn into_parts(
        self,
    ) -> Result<OpenAiImageGenerationRequestParts, OpenAiImageGenerationValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(OpenAiImageGenerationValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        if self.model.trim().is_empty() {
            return Err(OpenAiImageGenerationValidationError::BlankModel);
        }
        if self.prompt.trim().is_empty() {
            return Err(OpenAiImageGenerationValidationError::BlankPrompt);
        }
        validate_dimension("width", self.width)?;
        validate_dimension("height", self.height)?;
        if self.steps != SUPPORTED_IMAGE_STEPS {
            return Err(OpenAiImageGenerationValidationError::UnsupportedStepCount {
                actual_steps: self.steps,
            });
        }
        if self.guidance != SUPPORTED_IMAGE_GUIDANCE {
            return Err(OpenAiImageGenerationValidationError::UnsupportedGuidance {
                actual_guidance: self.guidance,
            });
        }
        if self.response_format != "b64_json" {
            return Err(
                OpenAiImageGenerationValidationError::UnsupportedResponseFormat {
                    response_format: self.response_format,
                },
            );
        }
        if self.n != 1 {
            return Err(
                OpenAiImageGenerationValidationError::UnsupportedImageCount {
                    actual_images: self.n,
                },
            );
        }

        Ok(OpenAiImageGenerationRequestParts {
            model: self.model,
            prompt: self.prompt,
            seed: self.seed,
            width: self.width,
            height: self.height,
            steps: self.steps,
            guidance: self.guidance,
            response_format: OpenAiImageGenerationResponseFormat::Base64Json,
            image_count: self.n,
        })
    }
}

/// Validated image request data ready for supervisor translation.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiImageGenerationRequestParts {
    pub model: String,
    pub prompt: String,
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    pub response_format: OpenAiImageGenerationResponseFormat,
    pub image_count: u32,
}

/// Output encoding admitted by the initial local image endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiImageGenerationResponseFormat {
    Base64Json,
}

/// A request rejected before image-model queue admission.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OpenAiImageGenerationValidationError {
    /// The caller supplied a field outside the supported request contract.
    #[error("request field '{field_name}' is unknown")]
    UnknownField { field_name: String },
    /// Model selection cannot resolve an empty identifier.
    #[error("model must not be blank")]
    BlankModel,
    /// Text conditioning requires visible prompt content.
    #[error("prompt must not be blank")]
    BlankPrompt,
    /// Native image geometry is bounded and aligned before expensive admission.
    #[error(
        "{parameter_name} must be a multiple of 16 in the {minimum_pixels}..={maximum_pixels} pixel range, received {actual_pixels}"
    )]
    UnsupportedDimension {
        parameter_name: &'static str,
        actual_pixels: u32,
        minimum_pixels: u32,
        maximum_pixels: u32,
    },
    /// The initial native profile has one qualified diffusion schedule.
    #[error("image generation supports exactly 4 steps, received {actual_steps}")]
    UnsupportedStepCount { actual_steps: u32 },
    /// The initial native profile has one qualified guidance setting.
    #[error("image generation supports guidance 1.0 only, received {actual_guidance}")]
    UnsupportedGuidance { actual_guidance: f32 },
    /// Only inline base64 JSON preserves the initial endpoint's local transport contract.
    #[error("image response format '{response_format}' is unsupported")]
    UnsupportedResponseFormat { response_format: String },
    /// The initial endpoint executes and returns one image per request.
    #[error("image generation supports exactly one image, received {actual_images}")]
    UnsupportedImageCount { actual_images: u32 },
}

fn default_image_count() -> u32 {
    1
}

fn validate_dimension(
    parameter_name: &'static str,
    actual_pixels: u32,
) -> Result<(), OpenAiImageGenerationValidationError> {
    let is_supported = (MIN_OPENAI_IMAGE_DIMENSION_PIXELS..=MAX_OPENAI_IMAGE_DIMENSION_PIXELS)
        .contains(&actual_pixels)
        && actual_pixels % IMAGE_DIMENSION_MULTIPLE_PIXELS == 0;
    if is_supported {
        return Ok(());
    }
    Err(OpenAiImageGenerationValidationError::UnsupportedDimension {
        parameter_name,
        actual_pixels,
        minimum_pixels: MIN_OPENAI_IMAGE_DIMENSION_PIXELS,
        maximum_pixels: MAX_OPENAI_IMAGE_DIMENSION_PIXELS,
    })
}
