//! A scripted speech answer for daemon lifecycle tests: two short segments,
//! then completion and finalization, in the real worker's event order. No
//! model runs, so the samples are a fixed ramp, not speech.

use astronomical_ipc_protocol::{
    ProtocolError, ProtocolWriter, SpeechResultMetadata, SpokenAudioSegment, WorkerCommand,
    WorkerEvent,
};
use tokio::io::AsyncWrite;

const SCRIPTED_SEGMENT_SAMPLE_COUNT: usize = 4;

/// Answers one speech command; any other command is left unanswered.
pub(super) async fn answer_speech_command<WriteTransport>(
    event_writer: &mut ProtocolWriter<WriteTransport>,
    worker_command: WorkerCommand,
) -> Result<(), ProtocolError>
where
    WriteTransport: AsyncWrite + Unpin,
{
    let WorkerCommand::GenerateSpeech(speech_command) = worker_command else {
        return Ok(());
    };
    let request_id = speech_command.request_id;
    for completed_segments in 1..=2_u32 {
        event_writer
            .send_event(&WorkerEvent::SpeechSegmentSpoken {
                request_id,
                audio_segment: SpokenAudioSegment {
                    pcm16_bytes: vec![0x10; SCRIPTED_SEGMENT_SAMPLE_COUNT * 2],
                },
                completed_segments,
                total_segments: 2,
                was_cut_short: false,
                elapsed_millis: u64::from(completed_segments),
                mlx_memory_snapshot: None,
            })
            .await?;
    }
    event_writer
        .send_event(&WorkerEvent::SpeechCompleted {
            request_id,
            result_metadata: SpeechResultMetadata {
                sample_rate_hertz: 24_000,
                sample_count: (SCRIPTED_SEGMENT_SAMPLE_COUNT * 2) as u64,
                segment_count: 2,
                cut_short_segment_count: 0,
                voice_name: speech_command
                    .voice_name
                    .unwrap_or_else(|| "af_heart".to_owned()),
                seed: speech_command.seed,
            },
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::SpeechFinalized {
            request_id,
            elapsed_millis: 3,
            mlx_memory_snapshot: None,
        })
        .await
}
