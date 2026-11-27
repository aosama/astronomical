mod admission;
mod chat;
mod daemon_side;
mod decision_training;
mod decisions;
mod embeddings;
mod image;
mod live_memory_limit_journeys;
mod speech;

/// Skips load-progress reports, which every autoregressive load now sends ahead of
/// its readiness event; the chat lifecycle tests cover their place in the stream.
async fn next_event_skipping_load_progress<ReadTransport>(
    daemon_reader: &mut ipc_protocol::ProtocolReader<ReadTransport>,
) -> ipc_protocol::WorkerEvent
where
    ReadTransport: tokio::io::AsyncRead + Unpin,
{
    loop {
        let worker_event = daemon_reader
            .next_event()
            .await
            .expect("the worker should write a valid event")
            .expect("the worker transport should remain open");
        if !matches!(
            worker_event,
            ipc_protocol::WorkerEvent::ModelLoadInFlight { .. }
        ) {
            return worker_event;
        }
    }
}
