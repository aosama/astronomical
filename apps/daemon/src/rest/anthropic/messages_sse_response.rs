//! Turns the stream encoder's events into the server-sent-events response
//! body: one SSE frame per event, named by the event's own type.

use std::collections::VecDeque;

use astronomical_rest_contract::AnthropicMessageStreamEvent;
use axum::response::{IntoResponse, Response, sse::Event, sse::Sse};
use futures_util::stream;
use tokio::sync::mpsc::Receiver;

use super::messages_stream::AnthropicMessagesStreamEncoder;
use crate::{ChatGenerationStreamErrorCode, ChatGenerationStreamEvent};

pub(crate) fn create_streaming_response(
    stream_event_receiver: Receiver<ChatGenerationStreamEvent>,
    stream_encoder: AnthropicMessagesStreamEncoder,
) -> Response {
    let initial_events = stream_encoder.initial_events();
    let message_event_stream = stream::unfold(
        (stream_event_receiver, stream_encoder, initial_events),
        next_serialized_stream_event,
    );
    Sse::new(message_event_stream).into_response()
}

type MessageStreamState = (
    Receiver<ChatGenerationStreamEvent>,
    AnthropicMessagesStreamEncoder,
    VecDeque<AnthropicMessageStreamEvent>,
);

/// Drains pending encoded events before reading the next worker event, and
/// ends the stream once the encoder has emitted its terminal event. A closed
/// worker channel becomes a worker-unavailable error event so a client never
/// sees a stream that stops without `message_stop` or `error`.
async fn next_serialized_stream_event(
    (mut stream_event_receiver, mut stream_encoder, mut pending_events): MessageStreamState,
) -> Option<(Result<Event, axum::Error>, MessageStreamState)> {
    loop {
        if let Some(message_stream_event) = pending_events.pop_front() {
            let serialized_event = serialize_stream_event(&message_stream_event);
            return Some((
                serialized_event,
                (stream_event_receiver, stream_encoder, pending_events),
            ));
        }
        if stream_encoder.is_terminal() {
            return None;
        }
        let stream_event = match stream_event_receiver.recv().await {
            Some(stream_event) => stream_event,
            None => {
                ChatGenerationStreamEvent::Error(ChatGenerationStreamErrorCode::WorkerUnavailable)
            }
        };
        pending_events.extend(stream_encoder.encode(stream_event));
    }
}

fn serialize_stream_event(
    message_stream_event: &AnthropicMessageStreamEvent,
) -> Result<Event, axum::Error> {
    let serialized_payload =
        serde_json::to_string(message_stream_event).map_err(axum::Error::new)?;
    Ok(Event::default()
        .event(message_stream_event.event_type())
        .data(serialized_payload))
}
