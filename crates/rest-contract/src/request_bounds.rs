//! The bounds every OpenAI-compatible generation request enforces alike: the
//! image-count and inline-media budgets, the generated-token range and the
//! sampling ranges. Chat Completions and Responses each wrote these checks and
//! their error variants before; both now report them through
//! [`RequestBoundsError`], wrapped transparently, so a caller reads the same
//! words from either endpoint.
//!
//! The Anthropic surface keeps its own errors: its messages are worded for its
//! own protocol.

use thiserror::Error;

use crate::request_budget::{
    MAX_IMAGE_PARTS_PER_REQUEST, MAX_INLINE_MEDIA_BYTES, MAX_OUTPUT_TOKENS,
};

/// A request outside one of the shared bounds.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RequestBoundsError {
    /// The request carried more image parts than the portable image-count budget.
    #[error(
        "the request carries {actual_image_part_count} image parts, exceeding the {maximum_image_part_count}-image budget by {excess_image_part_count} (mirrors the 100-image-per-request limit Anthropic documents for 200k-context models)"
    )]
    ImagePartCountExceedsBudget {
        actual_image_part_count: usize,
        maximum_image_part_count: usize,
        excess_image_part_count: usize,
    },
    /// The request's total decoded inline media exceeded the portable media budget.
    #[error(
        "the request carries {actual_inline_media_bytes} decoded image bytes, exceeding the {maximum_inline_media_bytes}-byte inline-media budget by {excess_inline_media_bytes} bytes (mirrors the 20 MB total inline request size of the Gemini API)"
    )]
    InlineMediaExceedsBudget {
        actual_inline_media_bytes: usize,
        maximum_inline_media_bytes: usize,
        excess_inline_media_bytes: usize,
    },
    /// The output token budget was zero or too large for the worker representation.
    #[error(
        "output token count is {actual_output_tokens}, outside the 1..={maximum_output_tokens} token range"
    )]
    OutputTokenCountOutOfRange {
        actual_output_tokens: u32,
        maximum_output_tokens: u32,
    },
    /// A sampling setting fell outside its OpenAI-compatible range.
    #[error("{parameter_name} is outside the supported range {minimum}..={maximum}")]
    SamplingParameterOutOfRange {
        parameter_name: &'static str,
        minimum: String,
        maximum: String,
    },
}

impl RequestBoundsError {
    /// Whether this is a request-size rejection (HTTP 413 under the portable
    /// request budget) rather than a malformed-request one (HTTP 400).
    #[must_use]
    pub fn is_request_too_large(&self) -> bool {
        matches!(
            self,
            Self::ImagePartCountExceedsBudget { .. } | Self::InlineMediaExceedsBudget { .. }
        )
    }
}

pub(crate) fn validate_image_part_count(image_part_count: usize) -> Result<(), RequestBoundsError> {
    if image_part_count > MAX_IMAGE_PARTS_PER_REQUEST {
        return Err(RequestBoundsError::ImagePartCountExceedsBudget {
            actual_image_part_count: image_part_count,
            maximum_image_part_count: MAX_IMAGE_PARTS_PER_REQUEST,
            excess_image_part_count: image_part_count - MAX_IMAGE_PARTS_PER_REQUEST,
        });
    }
    Ok(())
}

pub(crate) fn validate_inline_media_total(
    decoded_inline_media_bytes: usize,
) -> Result<(), RequestBoundsError> {
    if decoded_inline_media_bytes > MAX_INLINE_MEDIA_BYTES {
        return Err(RequestBoundsError::InlineMediaExceedsBudget {
            actual_inline_media_bytes: decoded_inline_media_bytes,
            maximum_inline_media_bytes: MAX_INLINE_MEDIA_BYTES,
            excess_inline_media_bytes: decoded_inline_media_bytes - MAX_INLINE_MEDIA_BYTES,
        });
    }
    Ok(())
}

/// The generated-token budget must be at least one token and fit the worker's
/// representation.
pub(crate) fn validate_output_token_count(
    actual_output_tokens: u32,
) -> Result<(), RequestBoundsError> {
    if actual_output_tokens == 0 || actual_output_tokens > MAX_OUTPUT_TOKENS {
        return Err(RequestBoundsError::OutputTokenCountOutOfRange {
            actual_output_tokens,
            maximum_output_tokens: MAX_OUTPUT_TOKENS,
        });
    }
    Ok(())
}

/// An absent setting is fine; a present one must be finite and in range.
pub(crate) fn validate_sampling_parameter(
    parameter_name: &'static str,
    parameter_value: Option<f32>,
    minimum: f32,
    maximum: f32,
) -> Result<(), RequestBoundsError> {
    let Some(parameter_value) = parameter_value else {
        return Ok(());
    };
    if parameter_value.is_finite() && parameter_value >= minimum && parameter_value <= maximum {
        return Ok(());
    }
    Err(RequestBoundsError::SamplingParameterOutOfRange {
        parameter_name,
        minimum: minimum.to_string(),
        maximum: maximum.to_string(),
    })
}

#[cfg(test)]
#[path = "request_bounds_tests.rs"]
mod request_bounds_tests;
