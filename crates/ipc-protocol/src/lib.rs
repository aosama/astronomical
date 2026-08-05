#![forbid(unsafe_code)]

mod chat_generation;
mod chat_generation_validation;
mod message_codec;
mod protocol_error;
mod protocol_message;
mod protocol_reader;
mod protocol_writer;

pub use chat_generation::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ChatGenerationSettings, ChatImageInput, ChatMessage, ChatModelCapabilities, ChatToolChoice,
    ChatToolDefinition,
};
pub use chat_generation_validation::ChatGenerationValidationError;
pub use message_codec::{decode_command, decode_event, encode_command, encode_event};
pub use protocol_error::ProtocolError;
pub use protocol_message::{
    ExpertMemoryMode, MAX_IPC_FRAME_BYTES, MlxMemorySnapshotSource, MtpRuntimeState, RequestId,
    WorkerCommand, WorkerEvent, WorkerLogLevel, WorkerMlxMemorySnapshot,
    WorkerPrefillChunckSizingPolicy, WorkerStartupConfiguration,
};
pub use protocol_reader::ProtocolReader;
pub use protocol_writer::ProtocolWriter;
