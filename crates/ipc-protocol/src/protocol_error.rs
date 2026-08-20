use std::io;

use thiserror::Error;

use crate::{
    ChatGenerationValidationError, ImageGenerationCompletionValidationError,
    ImageGenerationValidationError, WorkerModelCapabilitiesValidationError,
};

/// Errors raised while serializing, transmitting, or deserializing IPC messages.
#[derive(Debug, Error)]
pub enum ProtocolError {
    /// A decoded chat command violated the worker trust-boundary contract.
    #[error("received an invalid chat-generation command")]
    InvalidChatGenerationCommand(#[source] ChatGenerationValidationError),

    /// A decoded image command violated the worker trust-boundary contract.
    #[error("received an invalid image-generation command")]
    InvalidImageGenerationCommand(#[source] ImageGenerationValidationError),

    /// A worker advertised an impossible or empty capability contract.
    #[error("received invalid worker model capabilities")]
    InvalidWorkerModelCapabilities(#[source] WorkerModelCapabilitiesValidationError),

    /// Image completion bytes or metadata did not describe a protocol-valid PNG outcome.
    #[error("received an invalid image-generation completion")]
    InvalidImageGenerationCompletion(#[source] ImageGenerationCompletionValidationError),

    /// The serialized message cannot fit inside one bounded IPC frame.
    #[error(
        "IPC message is {actual_message_bytes} bytes, exceeding the {maximum_message_bytes}-byte limit"
    )]
    OutgoingMessageTooLarge {
        /// The number of serialized message bytes.
        actual_message_bytes: usize,
        /// The permitted message size in bytes.
        maximum_message_bytes: usize,
    },

    /// A message supplied by a message-oriented transport exceeded the frame cap.
    #[error(
        "received IPC message is {actual_message_bytes} bytes, exceeding the {maximum_message_bytes}-byte limit"
    )]
    IncomingMessageTooLarge {
        /// The number of received message bytes.
        actual_message_bytes: usize,
        /// The permitted message size in bytes.
        maximum_message_bytes: usize,
    },

    /// Reading a length-delimited frame failed.
    #[error("failed to read an IPC frame")]
    ReadFrame(#[source] io::Error),

    /// Writing a length-delimited frame failed.
    #[error("failed to write an IPC frame")]
    WriteFrame(#[source] io::Error),

    /// Serializing a typed message into JSON failed.
    #[error("failed to serialize an IPC message")]
    SerializeMessage(#[source] serde_json::Error),

    /// Deserializing a JSON frame into a typed message failed.
    #[error("failed to deserialize an IPC message")]
    DeserializeMessage(#[source] serde_json::Error),
}
