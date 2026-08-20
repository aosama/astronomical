use serde::{Serialize, de::DeserializeOwned};

use crate::{MAX_IPC_FRAME_BYTES, ProtocolError, WorkerCommand, WorkerEvent};

/// Serializes one bounded command sent to the inference worker.
pub fn encode_command(worker_command: &WorkerCommand) -> Result<Vec<u8>, ProtocolError> {
    encode_message(worker_command)
}

/// Deserializes one bounded command received by the inference worker.
pub fn decode_command(serialized_command: &[u8]) -> Result<WorkerCommand, ProtocolError> {
    decode_message(serialized_command)
}

/// Serializes one bounded event emitted by the inference worker.
pub fn encode_event(worker_event: &WorkerEvent) -> Result<Vec<u8>, ProtocolError> {
    encode_message(worker_event)
}

/// Deserializes one bounded event received from the inference worker.
pub fn decode_event(serialized_event: &[u8]) -> Result<WorkerEvent, ProtocolError> {
    let worker_event = decode_message(serialized_event)?;
    match &worker_event {
        WorkerEvent::Ready { capabilities, .. }
        | WorkerEvent::ModelSwapped { capabilities, .. } => capabilities
            .validate()
            .map_err(ProtocolError::InvalidWorkerModelCapabilities)?,
        WorkerEvent::ImageGenerationCompleted {
            generated_image,
            result_metadata,
            ..
        } => generated_image
            .validate_completion(result_metadata)
            .map_err(ProtocolError::InvalidImageGenerationCompletion)?,
        _ => {}
    }
    Ok(worker_event)
}

fn encode_message<Message>(message: &Message) -> Result<Vec<u8>, ProtocolError>
where
    Message: Serialize,
{
    let serialized_message =
        serde_json::to_vec(message).map_err(ProtocolError::SerializeMessage)?;
    if serialized_message.len() > MAX_IPC_FRAME_BYTES {
        return Err(ProtocolError::OutgoingMessageTooLarge {
            actual_message_bytes: serialized_message.len(),
            maximum_message_bytes: MAX_IPC_FRAME_BYTES,
        });
    }

    Ok(serialized_message)
}

fn decode_message<Message>(serialized_message: &[u8]) -> Result<Message, ProtocolError>
where
    Message: DeserializeOwned,
{
    if serialized_message.len() > MAX_IPC_FRAME_BYTES {
        return Err(ProtocolError::IncomingMessageTooLarge {
            actual_message_bytes: serialized_message.len(),
            maximum_message_bytes: MAX_IPC_FRAME_BYTES,
        });
    }

    serde_json::from_slice(serialized_message).map_err(ProtocolError::DeserializeMessage)
}
