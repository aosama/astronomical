//! Typed image-generation requests, outputs, capabilities, and worker-boundary validation.

mod png_validation;

use std::io::Cursor;

use image::{ColorType, ImageDecoder, ImageError, Limits, codecs::png::PngDecoder};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ChatModelCapabilities, MAX_IPC_FRAME_BYTES, RequestId};

use self::png_validation::validate_png_structure;

const MINIMUM_IMAGE_DIMENSION_PIXELS: u32 = 64;
const MAXIMUM_IMAGE_DIMENSION_PIXELS: u32 = 16_384;
const IMAGE_DIMENSION_MULTIPLE_PIXELS: u32 = 8;
const MINIMUM_IMAGE_GENERATION_STEPS: u16 = 1;
const MAXIMUM_IMAGE_GENERATION_STEPS: u16 = 1_000;
const MAXIMUM_IMAGE_GUIDANCE_THOUSANDTHS: u32 = 100_000;
const PNG_MIME_TYPE: &str = "image/png";
// A completion must not expand beyond the transport budget into a much larger memory owner.
const MAXIMUM_DECODED_RGB_BYTES: usize = MAX_IPC_FRAME_BYTES;

/// One validated text-to-image request sent to the local inference worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationCommand {
    pub request_id: RequestId,
    pub model: String,
    pub prompt: String,
    pub settings: ImageGenerationSettings,
}

impl ImageGenerationCommand {
    /// Independently validates image input after it crosses the worker trust boundary.
    pub fn validate(&self) -> Result<(), ImageGenerationValidationError> {
        if self.model.trim().is_empty() {
            return Err(ImageGenerationValidationError::EmptyModelId);
        }
        if self.prompt.trim().is_empty() {
            return Err(ImageGenerationValidationError::EmptyPrompt);
        }
        self.settings.validate()
    }
}

/// Bounded image dimensions, diffusion controls, and deterministic seed.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationSettings {
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub steps: u16,
    /// Classifier-free guidance scale represented in thousandths to keep JSON deterministic.
    pub guidance_thousandths: u32,
    pub seed: u64,
}

impl ImageGenerationSettings {
    /// Enforces protocol-wide safety bounds before model-specific capability checks.
    pub fn validate(&self) -> Result<(), ImageGenerationValidationError> {
        if !(MINIMUM_IMAGE_DIMENSION_PIXELS..=MAXIMUM_IMAGE_DIMENSION_PIXELS)
            .contains(&self.width_pixels)
        {
            return Err(ImageGenerationValidationError::WidthOutOfRange {
                actual_width_pixels: self.width_pixels,
                minimum_width_pixels: MINIMUM_IMAGE_DIMENSION_PIXELS,
                maximum_width_pixels: MAXIMUM_IMAGE_DIMENSION_PIXELS,
            });
        }
        if !self
            .width_pixels
            .is_multiple_of(IMAGE_DIMENSION_MULTIPLE_PIXELS)
        {
            return Err(ImageGenerationValidationError::WidthNotAligned {
                actual_width_pixels: self.width_pixels,
                required_multiple_pixels: IMAGE_DIMENSION_MULTIPLE_PIXELS,
            });
        }
        if !(MINIMUM_IMAGE_DIMENSION_PIXELS..=MAXIMUM_IMAGE_DIMENSION_PIXELS)
            .contains(&self.height_pixels)
        {
            return Err(ImageGenerationValidationError::HeightOutOfRange {
                actual_height_pixels: self.height_pixels,
                minimum_height_pixels: MINIMUM_IMAGE_DIMENSION_PIXELS,
                maximum_height_pixels: MAXIMUM_IMAGE_DIMENSION_PIXELS,
            });
        }
        if !self
            .height_pixels
            .is_multiple_of(IMAGE_DIMENSION_MULTIPLE_PIXELS)
        {
            return Err(ImageGenerationValidationError::HeightNotAligned {
                actual_height_pixels: self.height_pixels,
                required_multiple_pixels: IMAGE_DIMENSION_MULTIPLE_PIXELS,
            });
        }
        if !(MINIMUM_IMAGE_GENERATION_STEPS..=MAXIMUM_IMAGE_GENERATION_STEPS).contains(&self.steps)
        {
            return Err(ImageGenerationValidationError::StepsOutOfRange {
                actual_steps: self.steps,
                minimum_steps: MINIMUM_IMAGE_GENERATION_STEPS,
                maximum_steps: MAXIMUM_IMAGE_GENERATION_STEPS,
            });
        }
        if self.guidance_thousandths > MAXIMUM_IMAGE_GUIDANCE_THOUSANDTHS {
            return Err(ImageGenerationValidationError::GuidanceOutOfRange {
                actual_guidance_thousandths: self.guidance_thousandths,
                maximum_guidance_thousandths: MAXIMUM_IMAGE_GUIDANCE_THOUSANDTHS,
            });
        }
        Ok(())
    }
}

/// One generated encoded image; JSON uses base64 rather than an integer array.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedImage {
    pub mime_type: String,
    #[serde(with = "crate::base64_bytes")]
    pub encoded_bytes: Vec<u8>,
}

impl GeneratedImage {
    /// Fully decodes the PNG before trusting worker completion metadata.
    pub fn validate_completion(
        &self,
        result_metadata: &ImageGenerationResultMetadata,
    ) -> Result<(), ImageGenerationCompletionValidationError> {
        result_metadata
            .validate()
            .map_err(ImageGenerationCompletionValidationError::InvalidMetadata)?;
        if self.mime_type != PNG_MIME_TYPE {
            return Err(ImageGenerationCompletionValidationError::InvalidMimeType);
        }
        let validated_png = validate_png_structure(&self.encoded_bytes, MAXIMUM_DECODED_RGB_BYTES)?;
        if validated_png.width_pixels != result_metadata.width_pixels
            || validated_png.height_pixels != result_metadata.height_pixels
        {
            return Err(
                ImageGenerationCompletionValidationError::PngDimensionsMismatch {
                    encoded_width_pixels: validated_png.width_pixels,
                    encoded_height_pixels: validated_png.height_pixels,
                    metadata_width_pixels: result_metadata.width_pixels,
                    metadata_height_pixels: result_metadata.height_pixels,
                },
            );
        }
        let mut decoder_limits = Limits::default();
        decoder_limits.max_image_width = Some(MAXIMUM_IMAGE_DIMENSION_PIXELS);
        decoder_limits.max_image_height = Some(MAXIMUM_IMAGE_DIMENSION_PIXELS);
        decoder_limits.max_alloc = Some(
            u64::try_from(MAXIMUM_DECODED_RGB_BYTES)
                .map_err(|_| ImageGenerationCompletionValidationError::PngDecodeResourceLimit)?,
        );
        let png_decoder =
            PngDecoder::with_limits(Cursor::new(self.encoded_bytes.as_slice()), decoder_limits)
                .map_err(map_png_decode_error)?;
        if png_decoder.color_type() != ColorType::Rgb8 {
            return Err(ImageGenerationCompletionValidationError::NonRgb8Png);
        }
        if png_decoder.total_bytes()
            != u64::try_from(validated_png.decoded_rgb_byte_count)
                .map_err(|_| ImageGenerationCompletionValidationError::PngDecodeResourceLimit)?
        {
            return Err(ImageGenerationCompletionValidationError::InvalidPngEncoding);
        }
        let mut decoded_rgb_bytes = Vec::new();
        decoded_rgb_bytes
            .try_reserve_exact(validated_png.decoded_rgb_byte_count)
            .map_err(|_| ImageGenerationCompletionValidationError::PngDecodeResourceLimit)?;
        decoded_rgb_bytes.resize(validated_png.decoded_rgb_byte_count, 0);
        png_decoder
            .read_image(&mut decoded_rgb_bytes)
            .map_err(map_png_decode_error)
    }
}

fn map_png_decode_error(error: ImageError) -> ImageGenerationCompletionValidationError {
    if matches!(error, ImageError::Limits(_)) {
        ImageGenerationCompletionValidationError::PngDecodeResourceLimit
    } else {
        ImageGenerationCompletionValidationError::InvalidPngEncoding
    }
}

/// Reproducibility and timing facts for one completed image.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationResultMetadata {
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub steps: u16,
    pub guidance_thousandths: u32,
    pub seed: u64,
    pub elapsed_millis: u64,
}

impl ImageGenerationResultMetadata {
    /// Completion facts must obey the same protocol bounds as the accepted request.
    pub fn validate(&self) -> Result<(), ImageGenerationValidationError> {
        ImageGenerationSettings {
            width_pixels: self.width_pixels,
            height_pixels: self.height_pixels,
            steps: self.steps,
            guidance_thousandths: self.guidance_thousandths,
            seed: self.seed,
        }
        .validate()
    }
}

/// Worker execution phase used for user-visible image progress.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationPhase {
    Preparing,
    EncodingPrompt,
    Denoising,
    Decoding,
    EncodingImage,
}

/// A request-scoped image failure that leaves the worker protocol responsive.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationFailureReason {
    InvalidRequest { reason: String },
    ModelDoesNotSupportImageGeneration,
    EngineBusy,
    EncodingFailed { reason: String },
    FatalExecution { reason: String },
    Cancelled,
}

impl ImageGenerationFailureReason {
    #[must_use]
    pub fn invalid_request(reason: impl Into<String>) -> Self {
        Self::InvalidRequest {
            reason: reason.into(),
        }
    }
}

/// Image limits advertised by one loaded worker model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationCapabilities {
    pub minimum_width_pixels: u32,
    pub maximum_width_pixels: u32,
    pub minimum_height_pixels: u32,
    pub maximum_height_pixels: u32,
    pub dimension_multiple_pixels: u32,
    pub maximum_steps: u16,
    pub maximum_guidance_thousandths: u32,
    pub output_mime_types: Vec<String>,
}

impl ImageGenerationCapabilities {
    /// Rejects capability advertisements that cannot safely constrain a request.
    pub fn validate(&self) -> Result<(), WorkerModelCapabilitiesValidationError> {
        if self.dimension_multiple_pixels == 0 {
            return Err(WorkerModelCapabilitiesValidationError::ZeroImageDimensionAlignment);
        }
        if self.minimum_width_pixels > self.maximum_width_pixels {
            return Err(
                WorkerModelCapabilitiesValidationError::InvertedImageWidthBounds {
                    minimum_width_pixels: self.minimum_width_pixels,
                    maximum_width_pixels: self.maximum_width_pixels,
                },
            );
        }
        if self.minimum_height_pixels > self.maximum_height_pixels {
            return Err(
                WorkerModelCapabilitiesValidationError::InvertedImageHeightBounds {
                    minimum_height_pixels: self.minimum_height_pixels,
                    maximum_height_pixels: self.maximum_height_pixels,
                },
            );
        }
        if self.output_mime_types.is_empty()
            || self
                .output_mime_types
                .iter()
                .any(|mime_type| mime_type.trim().is_empty())
        {
            return Err(WorkerModelCapabilitiesValidationError::EmptyImageOutputMimeType);
        }
        Ok(())
    }
}

/// Independently represents the chat and image surfaces exposed by one model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelCapabilities {
    pub chat: Option<ChatModelCapabilities>,
    pub image_generation: Option<ImageGenerationCapabilities>,
}

impl WorkerModelCapabilities {
    #[must_use]
    pub fn image_generation(image_generation: ImageGenerationCapabilities) -> Self {
        Self {
            chat: None,
            image_generation: Some(image_generation),
        }
    }

    #[must_use]
    pub fn chat_and_image(
        chat: ChatModelCapabilities,
        image_generation: ImageGenerationCapabilities,
    ) -> Self {
        Self {
            chat: Some(chat),
            image_generation: Some(image_generation),
        }
    }

    /// A loaded model must advertise at least one usable operation surface.
    pub fn validate(&self) -> Result<(), WorkerModelCapabilitiesValidationError> {
        if self.chat.is_none() && self.image_generation.is_none() {
            return Err(WorkerModelCapabilitiesValidationError::NoCapabilities);
        }
        if let Some(image_generation) = &self.image_generation {
            image_generation.validate()?;
        }
        Ok(())
    }
}

impl From<ChatModelCapabilities> for WorkerModelCapabilities {
    fn from(chat: ChatModelCapabilities) -> Self {
        Self {
            chat: Some(chat),
            image_generation: None,
        }
    }
}

/// A bounded semantic validation failure in one image-generation command.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ImageGenerationValidationError {
    #[error("model ID must not be empty")]
    EmptyModelId,
    #[error("image prompt must not be empty")]
    EmptyPrompt,
    #[error(
        "image width {actual_width_pixels} is outside {minimum_width_pixels}..={maximum_width_pixels} pixels"
    )]
    WidthOutOfRange {
        actual_width_pixels: u32,
        minimum_width_pixels: u32,
        maximum_width_pixels: u32,
    },
    #[error(
        "image width {actual_width_pixels} must be a multiple of {required_multiple_pixels} pixels"
    )]
    WidthNotAligned {
        actual_width_pixels: u32,
        required_multiple_pixels: u32,
    },
    #[error(
        "image height {actual_height_pixels} is outside {minimum_height_pixels}..={maximum_height_pixels} pixels"
    )]
    HeightOutOfRange {
        actual_height_pixels: u32,
        minimum_height_pixels: u32,
        maximum_height_pixels: u32,
    },
    #[error(
        "image height {actual_height_pixels} must be a multiple of {required_multiple_pixels} pixels"
    )]
    HeightNotAligned {
        actual_height_pixels: u32,
        required_multiple_pixels: u32,
    },
    #[error("image step count {actual_steps} is outside {minimum_steps}..={maximum_steps}")]
    StepsOutOfRange {
        actual_steps: u16,
        minimum_steps: u16,
        maximum_steps: u16,
    },
    #[error(
        "image guidance {actual_guidance_thousandths} thousandths exceeds {maximum_guidance_thousandths}"
    )]
    GuidanceOutOfRange {
        actual_guidance_thousandths: u32,
        maximum_guidance_thousandths: u32,
    },
}

/// A malformed image completion received from the worker process.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ImageGenerationCompletionValidationError {
    #[error("image completion metadata is invalid")]
    InvalidMetadata(#[source] ImageGenerationValidationError),
    #[error("completed image MIME type must be exactly image/png")]
    InvalidMimeType,
    #[error("completed image is not a fully decodable PNG")]
    InvalidPngEncoding,
    #[error("completed PNG must use lossless 8-bit RGB pixels")]
    NonRgb8Png,
    #[error("completed PNG exceeds the bounded decode resource limit")]
    PngDecodeResourceLimit,
    #[error(
        "completed PNG dimensions {encoded_width_pixels}x{encoded_height_pixels} do not match metadata {metadata_width_pixels}x{metadata_height_pixels}"
    )]
    PngDimensionsMismatch {
        encoded_width_pixels: u32,
        encoded_height_pixels: u32,
        metadata_width_pixels: u32,
        metadata_height_pixels: u32,
    },
}

/// A bounded semantic failure in a worker capability advertisement.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum WorkerModelCapabilitiesValidationError {
    #[error("worker model must advertise at least one capability")]
    NoCapabilities,
    #[error("image dimension alignment must be positive")]
    ZeroImageDimensionAlignment,
    #[error(
        "image width bounds are inverted: minimum {minimum_width_pixels} exceeds maximum {maximum_width_pixels}"
    )]
    InvertedImageWidthBounds {
        minimum_width_pixels: u32,
        maximum_width_pixels: u32,
    },
    #[error(
        "image height bounds are inverted: minimum {minimum_height_pixels} exceeds maximum {maximum_height_pixels}"
    )]
    InvertedImageHeightBounds {
        minimum_height_pixels: u32,
        maximum_height_pixels: u32,
    },
    #[error("image output MIME types must contain only nonempty values")]
    EmptyImageOutputMimeType,
}
