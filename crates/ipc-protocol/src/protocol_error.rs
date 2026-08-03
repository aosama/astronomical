use std::io;

use thiserror::Error;

/// Errors raised while serializing, transmitting, or deserializing IPC messages.
#[derive(Debug, Error)]
pub enum ProtocolError {
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
