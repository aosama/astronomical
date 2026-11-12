//! Scripted answers for the requests a worker serves in one forward pass:
//! embeddings and decisions.
//!
//! Both follow the same completed-then-finalized event order and leave
//! nothing active behind them, so the fixtures answer them through one arm.

use astronomical_ipc_protocol::{ProtocolError, ProtocolWriter, WorkerCommand, WorkerEvent};
use tokio::io::AsyncWrite;

use super::scripted_worker_decisions;

/// Answers one embeddings or decisions command and finalizes it. A speech
/// command is answered in its own scripted module; any other command is not
/// a request these fixtures answer and is left unanswered.
pub(super) async fn answer_single_shot_command<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    single_shot_command: WorkerCommand,
) -> Result<(), ProtocolError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    match single_shot_command {
        WorkerCommand::GenerateDecisions(decisions_command) => {
            scripted_worker_decisions::answer_decisions_command(event_writer, decisions_command)
                .await
        }
        WorkerCommand::GenerateEmbeddings(embeddings_command) => {
            event_writer
                .send_event(&WorkerEvent::EmbeddingsCompleted {
                    request_id: embeddings_command.request_id,
                    embeddings: vec![vec![1.0, 0.0]; embeddings_command.inputs.len()],
                    input_token_counts: vec![1; embeddings_command.inputs.len()],
                    elapsed_millis: 1,
                })
                .await?;
            event_writer
                .send_event(&WorkerEvent::EmbeddingsFinalized {
                    request_id: embeddings_command.request_id,
                    elapsed_millis: 2,
                    mlx_memory_snapshot: None,
                })
                .await
        }
        other_command => {
            super::scripted_worker_speech::answer_speech_command(event_writer, other_command).await
        }
    }
}
