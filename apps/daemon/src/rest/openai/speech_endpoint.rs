//! The speech endpoint, `POST /v1/audio/speech`, in OpenAI's request shape,
//! with complete validation before shared queue admission.
//!
//! The response is the audio itself: a WAV file, or bare 16-bit mono
//! samples for `response_format: "pcm"`. How it was spoken travels in
//! headers, so a caller can reproduce it (the seed) and play bare samples
//! (the sample rate) without a second request.

use astronomical_config::SpeechModelCapabilities;
use astronomical_rest_contract::{OpenAiSpeechRequest, OpenAiSpeechRequestParts};
use astronomical_speech_contract::{
    AudioFormat, NATURAL_PACE_ONLY_REASON, SpeechFailure, unknown_voice_reason,
    wav_bytes_from_pcm16,
};
use axum::{
    Json,
    extract::State,
    http::{HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::{
    GenerationStartError, SpeechExecutionError, SpeechOutput,
    application::{ApplicationState, allocate_chat_request_id},
    rest::request_rejections::{
        generation_start_error_response, invalid_request_response,
        request_identifier_exhausted_response, worker_unavailable_response,
    },
    serving::speech_service::{
        SpeechAsked, SpeechModelResolutionError, build_speech_command, resolve_speech_model,
        speech_capabilities_of,
    },
};

const SEED_HEADER: &str = "x-astronomical-seed";
const VOICE_HEADER: &str = "x-astronomical-voice";
const SAMPLE_RATE_HEADER: &str = "x-astronomical-sample-rate";
const CUT_SHORT_SEGMENTS_HEADER: &str = "x-astronomical-segments-cut-short";

pub(crate) async fn create_speech(
    State(application_state): State<ApplicationState>,
    request_body_bytes: axum::body::Bytes,
) -> Response {
    let speech_request_parts = match parse_speech_request(request_body_bytes.as_ref()) {
        Ok(speech_request_parts) => speech_request_parts,
        Err(rejection_response) => return rejection_response,
    };
    let response_format = speech_request_parts.response_format;
    let configuration_transition_guard =
        application_state.configuration_transition_lock.lock().await;
    let (speech_command, request_identifier) =
        match prepared_speech_command(&application_state, speech_request_parts) {
            Ok(prepared_command) => prepared_command,
            Err(rejection_response) => return rejection_response,
        };
    let (admission_sender, mut admission_receiver) = tokio::sync::oneshot::channel();
    let mut speech_start_future = application_state
        .generation_executor
        .start_speech_generation_with_admission_signal(speech_command, admission_sender);
    // The configuration lock is held only until the request owns its place
    // in the queue, so a reload cannot slip between resolution and admission.
    let speech_start_result = tokio::select! {
        admission_result = &mut admission_receiver => {
            drop(configuration_transition_guard);
            match admission_result {
                Ok(()) => speech_start_future.await,
                Err(_) => Err(GenerationStartError::WorkerUnavailable),
            }
        }
        speech_start_result = &mut speech_start_future => {
            drop(configuration_transition_guard);
            speech_start_result
        }
    };
    let mut speech_result_receiver = match speech_start_result {
        Ok(speech_result_receiver) => speech_result_receiver,
        Err(start_error) => return generation_start_error_response(start_error),
    };
    match speech_result_receiver.recv().await {
        Some(Ok(speech_output)) => audio_response(speech_output, response_format),
        Some(Err(speech_error)) => {
            speech_execution_error_response(request_identifier, speech_error)
        }
        None => worker_unavailable_response(),
    }
}

/// Chooses the model, checks the request against it, and builds the worker
/// command, or answers why the request cannot be spoken.
#[allow(clippy::result_large_err)]
fn prepared_speech_command(
    application_state: &ApplicationState,
    speech_request_parts: OpenAiSpeechRequestParts,
) -> Result<(astronomical_ipc_protocol::SpeechCommand, u64), Response> {
    let worker_health_snapshot = application_state
        .generation_executor
        .worker_health_snapshot();
    if !worker_health_snapshot.status.is_ready() {
        return Err(worker_unavailable_response());
    }
    let discovered_models = application_state.discovered_models_snapshot();
    let resolved_model_id = resolve_speech_model(
        &discovered_models,
        worker_health_snapshot.ready_model_id.as_deref(),
        speech_request_parts.model.as_deref(),
    )
    .map_err(|resolution_error| model_resolution_rejection_response(&resolution_error))?;
    if let Some(rejection_response) = refusal_before_swap(
        speech_capabilities_of(&discovered_models, &resolved_model_id),
        &speech_request_parts,
    ) {
        return Err(rejection_response);
    }
    let Some(request_identifier) =
        allocate_chat_request_id(&application_state.next_chat_request_id)
    else {
        return Err(request_identifier_exhausted_response());
    };
    let speech_command = build_speech_command(
        request_identifier,
        &resolved_model_id,
        speech_asked(speech_request_parts),
    );
    Ok((speech_command, request_identifier))
}

/// Reads and checks the body, rejecting anything malformed before any state
/// is touched.
// The Err variant carries the fully built axum response; it is large but
// constructed once and moved once, so boxing it would only add indirection.
#[allow(clippy::result_large_err)]
fn parse_speech_request(request_body_bytes: &[u8]) -> Result<OpenAiSpeechRequestParts, Response> {
    let speech_request: OpenAiSpeechRequest =
        serde_json::from_slice(request_body_bytes).map_err(|json_error| {
            invalid_request_response(
                format!("request body is not a valid speech request: {json_error}"),
                None,
                "invalid_json",
            )
        })?;
    speech_request.into_parts().map_err(|validation_error| {
        invalid_request_response(
            validation_error.to_string(),
            validation_error.parameter_name(),
            "invalid_request",
        )
    })
}

/// Refuses a voice the chosen model lacks, or a pace it cannot change,
/// before the request can make that model resident.
fn refusal_before_swap(
    speech_capabilities: Option<&SpeechModelCapabilities>,
    speech_request_parts: &OpenAiSpeechRequestParts,
) -> Option<Response> {
    let speech_capabilities = speech_capabilities?;
    if let Some(voice_name) = &speech_request_parts.voice
        && !speech_capabilities.voice_names.contains(voice_name)
    {
        return Some(invalid_request_response(
            unknown_voice_reason(voice_name, &speech_capabilities.voice_names),
            Some("voice"),
            "voice_not_found",
        ));
    }
    if !speech_capabilities.supports_speed && speech_request_parts.speed != 1.0 {
        return Some(invalid_request_response(
            NATURAL_PACE_ONLY_REASON,
            Some("speed"),
            "speed_not_supported",
        ));
    }
    None
}

fn speech_asked(speech_request_parts: OpenAiSpeechRequestParts) -> SpeechAsked {
    SpeechAsked {
        text: speech_request_parts.input,
        voice_name: speech_request_parts.voice,
        speed: speech_request_parts.speed,
        seed: speech_request_parts.seed,
    }
}

/// The audio in the asked format, with how it was spoken in headers.
fn audio_response(speech_output: SpeechOutput, response_format: AudioFormat) -> Response {
    let result_metadata = speech_output.result_metadata;
    let audio_bytes = match response_format {
        AudioFormat::Pcm => speech_output.pcm16_bytes,
        AudioFormat::Wav => {
            match wav_bytes_from_pcm16(
                &speech_output.pcm16_bytes,
                result_metadata.sample_rate_hertz,
            ) {
                Ok(wav_bytes) => wav_bytes,
                Err(encoding_error) => {
                    // The worker is healthy; the daemon could not wrap what
                    // it returned, so this is the daemon's own failure.
                    tracing::error!(error = %encoding_error, "speech audio could not be wrapped as WAV");
                    return speech_failed_response();
                }
            }
        }
    };
    let mut audio_response = (StatusCode::OK, audio_bytes).into_response();
    let response_headers = audio_response.headers_mut();
    response_headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(response_format.mime_type()),
    );
    let described_headers = [
        (SEED_HEADER, result_metadata.seed.to_string()),
        (
            SAMPLE_RATE_HEADER,
            result_metadata.sample_rate_hertz.to_string(),
        ),
        (VOICE_HEADER, result_metadata.voice_name),
        (
            CUT_SHORT_SEGMENTS_HEADER,
            result_metadata.cut_short_segment_count.to_string(),
        ),
    ];
    for (header_name, header_text) in described_headers {
        if let Ok(header_value) = HeaderValue::from_str(&header_text) {
            response_headers.insert(HeaderName::from_static(header_name), header_value);
        }
    }
    audio_response
}

fn model_resolution_rejection_response(resolution_error: &SpeechModelResolutionError) -> Response {
    let error_code = match resolution_error {
        SpeechModelResolutionError::ModelNotFound => "model_not_found",
        SpeechModelResolutionError::ModelCannotSpeak => "model_capability_mismatch",
        SpeechModelResolutionError::NoSpeechModelInstalled => "no_speech_model",
        SpeechModelResolutionError::SeveralSpeechModelsInstalled { .. } => "speech_model_ambiguous",
    };
    invalid_request_response(resolution_error.to_string(), Some("model"), error_code)
}

fn speech_execution_error_response(
    request_identifier: u64,
    speech_error: SpeechExecutionError,
) -> Response {
    match speech_error {
        // The reason names the voice, pace or text at fault, and is already
        // bounded and free of local paths, so it is returned as written.
        SpeechExecutionError::WorkerFailure(SpeechFailure::InvalidRequest { reason }) => {
            invalid_request_response(reason, None, "invalid_speech_request")
        }
        SpeechExecutionError::WorkerFailure(SpeechFailure::EngineBusy) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::capacity_unavailable(
                    "the speech engine is busy",
                ),
            ),
        )
            .into_response(),
        SpeechExecutionError::DeadlineExceeded => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(
                astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                    "the speech request stopped making progress in the local worker",
                    Some("speech_timeout"),
                ),
            ),
        )
            .into_response(),
        SpeechExecutionError::WorkerFailure(
            failure @ (SpeechFailure::FatalExecution { .. } | SpeechFailure::Cancelled),
        ) => {
            tracing::error!(request_identifier, reason = %failure, "speech worker execution failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                        "the speech request failed in the local worker",
                        Some("speech_failed"),
                    ),
                ),
            )
                .into_response()
        }
        SpeechExecutionError::WorkerUnavailable => worker_unavailable_response(),
    }
}

fn speech_failed_response() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(
            astronomical_rest_contract::OpenAiErrorResponse::service_unavailable(
                "the speech request failed in the local worker",
                Some("speech_failed"),
            ),
        ),
    )
        .into_response()
}

#[cfg(test)]
#[path = "speech_endpoint_tests.rs"]
mod speech_endpoint_tests;
