//! Daemon-side helpers that send chat requests and cancellations over the IPC
//! pipe and describe the events the worker answers with.
use super::support::next_event;
use ipc_protocol::{
    ChatGenerationCompletionReason, ExpertMemoryMode, ProtocolReader, ProtocolWriter, RequestId,
    WorkerCommand, WorkerEvent,
};
use tokio::io::{AsyncRead, AsyncWrite};

pub(crate) async fn send_chat_request<WriteTransport>(
    daemon_writer: &mut ProtocolWriter<WriteTransport>,
    request_number: u64,
    seed: u64,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    daemon_writer
        .send_command(&WorkerCommand::Generate(super::support::chat_command(
            request_number,
            seed,
        )))
        .await
        .expect("the worker should receive a chat request");
}

pub(crate) async fn send_cancellation<WriteTransport>(
    daemon_writer: &mut ProtocolWriter<WriteTransport>,
    request_number: u64,
) where
    WriteTransport: AsyncWrite + Unpin,
{
    daemon_writer
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(request_number),
        })
        .await
        .expect("the worker should receive cancellation");
}

pub(crate) async fn expect_expert_memory_mode_changed<ReadTransport>(
    daemon_reader: &mut ProtocolReader<ReadTransport>,
    expected_expert_memory_mode: ExpertMemoryMode,
) where
    ReadTransport: AsyncRead + Unpin,
{
    assert_eq!(
        next_event(daemon_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: expected_expert_memory_mode,
        }
    );
}

/// The terminal event of a request that reported no astro_ssd_prompt_cache
/// activity and no reasoning tokens, which is what every scripted engine here
/// produces unless a test says otherwise.
pub(crate) fn plain_completed_event(
    request_number: u64,
    prompt_token_count: u32,
    generated_token_count: u16,
    reason: ChatGenerationCompletionReason,
) -> WorkerEvent {
    WorkerEvent::Completed {
        astro_ssd_prompt_cache_diagnostics: None,
        request_id: RequestId::new(request_number),
        prompt_token_count,
        generated_token_count,
        reasoning_token_count: 0,
        astro_ssd_prompt_cache_restored_token_count: 0,
        reason,
    }
}
