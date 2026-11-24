//! What a loaded model tells the daemon it can do, and the checks that
//! make an advertisement safe to route requests by.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ChatModelCapabilities, ImageGenerationCapabilities, WorkerEmbeddingCapabilities};

/// What one loaded model can be asked: chat, images, embeddings, decisions or speech.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelCapabilities {
    pub chat: Option<ChatModelCapabilities>,
    pub image_generation: Option<ImageGenerationCapabilities>,
    pub embeddings: Option<WorkerEmbeddingCapabilities>,
    pub decisions: Option<crate::WorkerDecisionCapabilities>,
    pub speech: Option<crate::WorkerSpeechCapabilities>,
}

impl WorkerModelCapabilities {
    #[must_use]
    pub fn embeddings(embedding: WorkerEmbeddingCapabilities) -> Self {
        Self {
            chat: None,
            image_generation: None,
            embeddings: Some(embedding),
            decisions: None,
            speech: None,
        }
    }

    #[must_use]
    pub fn decisions(decisions: crate::WorkerDecisionCapabilities) -> Self {
        Self {
            chat: None,
            image_generation: None,
            embeddings: None,
            decisions: Some(decisions),
            speech: None,
        }
    }

    #[must_use]
    pub fn speech(speech: crate::WorkerSpeechCapabilities) -> Self {
        Self {
            chat: None,
            image_generation: None,
            embeddings: None,
            decisions: None,
            speech: Some(speech),
        }
    }

    #[must_use]
    pub fn image_generation(image_generation: ImageGenerationCapabilities) -> Self {
        Self {
            chat: None,
            image_generation: Some(image_generation),
            embeddings: None,
            decisions: None,
            speech: None,
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
            embeddings: None,
            decisions: None,
            speech: None,
        }
    }

    /// A loaded model must advertise at least one usable operation surface.
    pub fn validate(&self) -> Result<(), WorkerModelCapabilitiesValidationError> {
        if self.advertises_no_surface() {
            return Err(WorkerModelCapabilitiesValidationError::NoCapabilities);
        }
        if let Some(image_generation) = &self.image_generation {
            image_generation.validate()?;
        }
        if let Some(embeddings) = &self.embeddings {
            embeddings.validate()?;
        }
        if let Some(speech) = &self.speech {
            speech.validate()?;
        }
        self.validate_decisions()
    }

    /// Whether the model offers nothing a request could use.
    fn advertises_no_surface(&self) -> bool {
        self.chat.is_none()
            && self.image_generation.is_none()
            && self.embeddings.is_none()
            && self.decisions.is_none()
            && self.speech.is_none()
    }

    /// A decisions surface must be able to hold at least one question.
    fn validate_decisions(&self) -> Result<(), WorkerModelCapabilitiesValidationError> {
        let Some(decisions) = &self.decisions else {
            return Ok(());
        };
        if decisions.is_usable() {
            return Ok(());
        }
        Err(
            WorkerModelCapabilitiesValidationError::DecisionTokenLimitTooSmall {
                actual_tokens_per_question: decisions.maximum_tokens_per_question,
            },
        )
    }
}

impl From<ChatModelCapabilities> for WorkerModelCapabilities {
    fn from(chat: ChatModelCapabilities) -> Self {
        Self {
            chat: Some(chat),
            image_generation: None,
            embeddings: None,
            decisions: None,
            speech: None,
        }
    }
}

/// A bounded semantic failure in a worker capability advertisement.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum WorkerModelCapabilitiesValidationError {
    #[error("worker model must advertise at least one capability")]
    NoCapabilities,
    #[error("embedding vector width must be positive")]
    ZeroEmbeddingVectorWidth,
    #[error("embedding context must hold at least two positions, got {actual_context_tokens}")]
    EmbeddingContextTooSmall { actual_context_tokens: u32 },
    #[error(
        "a decision model must read at least {} tokens per question, got {actual_tokens_per_question}",
        crate::MINIMUM_DECISION_TOKENS_PER_QUESTION
    )]
    DecisionTokenLimitTooSmall { actual_tokens_per_question: u32 },
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
    #[error("a speech model must speak at a positive sample rate")]
    ZeroSpeechSampleRate,
    #[error("a speech model must offer at least one voice, and its default among them")]
    MissingSpeechVoice,
}
