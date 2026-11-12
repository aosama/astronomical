//! Speech routing for the ephemeral daemon IPC service: resolves the model,
//! dispatches one speech command to the worker executor, and answers with
//! the audio in bounded frames, then one terminal frame.

use astronomical_ipc_protocol::{
    DaemonResponse, DaemonTransportError, MAXIMUM_SPEECH_AUDIO_FRAME_BYTES, SpeakGenerateRequest,
    StreamingResponseWriter,
};
use astronomical_speech_contract::SpeechFailure;

use crate::{
    DaemonIpcGenerationContext, GenerationStartError, SpeechExecutionError, SpeechOutput,
    application::allocate_chat_request_id,
    ipc::service::generation_start_rejection_reason,
    serving::speech_service::{
        SpeechAsked, build_speech_command, refuse_voice_or_pace_before_swap, resolve_speech_model,
        speech_capabilities_of,
    },
};

/// What one IPC speech request ended in.
enum SpeechOutcome {
    Spoken {
        model: String,
        speech_output: SpeechOutput,
    },
    Terminal(DaemonResponse),
}

/// Runs one IPC speech request: the audio frames, then exactly one terminal frame.
pub(crate) async fn run_speech_generation(
    generation_context: &DaemonIpcGenerationContext,
    speak_request: SpeakGenerateRequest,
    mut streaming_response_writer: StreamingResponseWriter,
) -> Result<(), DaemonTransportError> {
    let terminal_response = match speak(generation_context, speak_request).await {
        SpeechOutcome::Terminal(terminal_response) => terminal_response,
        SpeechOutcome::Spoken {
            model,
            speech_output,
        } => {
            for pcm16_frame in speech_output
                .pcm16_bytes
                .chunks(MAXIMUM_SPEECH_AUDIO_FRAME_BYTES)
            {
                streaming_response_writer
                    .send_response(&DaemonResponse::SpeechAudio {
                        pcm16_bytes: pcm16_frame.to_vec(),
                    })
                    .await?;
            }
            DaemonResponse::SpeechCompleted {
                model,
                metadata: speech_output.result_metadata,
            }
        }
    };
    streaming_response_writer
        .send_response(&terminal_response)
        .await?;
    streaming_response_writer.close().await
}

/// Chooses the model, refuses what it cannot speak, and waits for the worker.
async fn speak(
    generation_context: &DaemonIpcGenerationContext,
    speak_request: SpeakGenerateRequest,
) -> SpeechOutcome {
    let health_snapshot = generation_context.executor.worker_health_snapshot();
    let discovered_models = match generation_context.reloadable_config.as_ref() {
        Some(reloadable_config) => match reloadable_config.read() {
            Ok(resolved_runtime_config) => resolved_runtime_config.discovered_models.clone(),
            Err(_poisoned_lock) => Vec::new(),
        },
        None => Vec::new(),
    };
    let resolved_model_id = match resolve_speech_model(
        &discovered_models,
        health_snapshot.ready_model_id.as_deref(),
        speak_request.model.as_deref(),
    ) {
        Ok(resolved_model_id) => resolved_model_id,
        Err(resolution_error) => return rejected(resolution_error.to_string()),
    };
    if let Err(refusal) = refuse_voice_or_pace_before_swap(
        speech_capabilities_of(&discovered_models, &resolved_model_id),
        speak_request.voice_name.as_deref(),
        speak_request.speed,
    ) {
        return failed(SpeechFailure::invalid_request(refusal.reason()));
    }
    let Some(request_identifier) =
        allocate_chat_request_id(&generation_context.next_chat_request_id)
    else {
        return rejected("the local request identifier space is exhausted".to_owned());
    };
    let speech_command = build_speech_command(
        request_identifier,
        &resolved_model_id,
        SpeechAsked {
            text: speak_request.text,
            voice_name: speak_request.voice_name,
            speed: speak_request.speed,
            seed: speak_request.seed,
        },
    );
    await_speech(generation_context, speech_command, resolved_model_id).await
}

async fn await_speech(
    generation_context: &DaemonIpcGenerationContext,
    speech_command: astronomical_ipc_protocol::SpeechCommand,
    resolved_model_id: String,
) -> SpeechOutcome {
    let mut speech_result_receiver = match generation_context
        .executor
        .start_speech_generation(speech_command)
        .await
    {
        Ok(speech_result_receiver) => speech_result_receiver,
        Err(start_error) => return rejected(generation_start_rejection_reason(start_error)),
    };
    match speech_result_receiver.recv().await {
        Some(Ok(speech_output)) => SpeechOutcome::Spoken {
            model: resolved_model_id,
            speech_output,
        },
        Some(Err(SpeechExecutionError::WorkerFailure(reason))) => failed(reason),
        Some(Err(SpeechExecutionError::DeadlineExceeded)) => failed(
            SpeechFailure::fatal_execution("the speech request stopped making progress"),
        ),
        Some(Err(SpeechExecutionError::WorkerUnavailable)) => rejected(
            generation_start_rejection_reason(GenerationStartError::WorkerUnavailable),
        ),
        None => failed(SpeechFailure::fatal_execution(
            "the worker stream ended before the speech request completed",
        )),
    }
}

fn rejected(reason: String) -> SpeechOutcome {
    SpeechOutcome::Terminal(DaemonResponse::GenerationRejected { reason })
}

fn failed(reason: SpeechFailure) -> SpeechOutcome {
    SpeechOutcome::Terminal(DaemonResponse::SpeechFailed { reason })
}
