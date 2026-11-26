//! The stub daemon's per-connection loop: one scripted exchange sequence per
//! connection, answered request by request like the real daemon.
//!
//! Split from `stub_daemon.rs` so each request kind has a short answering
//! function of its own instead of one long accept-loop body.

use std::sync::{Arc, Mutex};

use ipc_protocol::{
    ChatGenerationSettings, ChatMessage, DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION,
    DaemonCatalogEntry, DaemonListedModel, DaemonRequest, DaemonResponse, ProtocolReader,
    ProtocolWriter,
};

use super::{
    StubDaemonConfig, next_scripted_load_report, next_scripted_progress_count,
    scripted_helper_model_frames, scripted_image_response, send_frames_and_close,
    wait_out_hung_status_request,
};
use crate::integration::support::stub_decisions::training_refusal;
use crate::integration::support::stub_download_status::scripted_download_status;
use crate::integration::support::stub_inventory::{StubCatalogEntry, StubInstalledModel};
use crate::integration::support::stub_model_control::stub_model_control_answer;

type StubConnectionWriter = ProtocolWriter<tokio::net::unix::OwnedWriteHalf>;

/// Everything the stub's connections share: the scripted behaviour plus the
/// counters that number requests across every connection, so a script reads
/// the same however many connections the client opens.
pub(super) struct StubDaemonState {
    pub(super) config: StubDaemonConfig,
    pub(super) resident_model_id: Option<String>,
    pub(super) catalog_entries_after_download: Option<Vec<StubCatalogEntry>>,
    pub(super) default_model_id: Mutex<Option<String>>,
    pub(super) download_job_index: Mutex<usize>,
    pub(super) download_started: Mutex<bool>,
    pub(super) model_load_report_index: Mutex<usize>,
    pub(super) request_progress_count_index: Mutex<usize>,
    pub(super) status_request_count: Mutex<usize>,
}

/// What a request leaves of the connection: the writer to answer the next
/// request with, or nothing once an answer ended the exchange. `close`
/// consumes the writer, so the ending kinds take it by value.
enum ConnectionFlow {
    KeepServing(StubConnectionWriter),
    Finished,
}

/// Answers requests on one connection until the client hangs up or a request
/// kind ends the exchange.
pub(super) async fn serve_stub_connection(
    connection_stream: tokio::net::UnixStream,
    state: Arc<StubDaemonState>,
) {
    let (read_half, write_half) = connection_stream.into_split();
    let mut protocol_reader = ProtocolReader::new(read_half);
    let mut protocol_writer = ProtocolWriter::new(write_half);
    loop {
        let Some(daemon_request) = protocol_reader
            .next_daemon_request()
            .await
            .expect("the stub daemon request read should not fail at the transport layer")
        else {
            return;
        };
        match answer_stub_request(&state, protocol_writer, daemon_request).await {
            ConnectionFlow::KeepServing(writer_for_next_request) => {
                protocol_writer = writer_for_next_request;
            }
            ConnectionFlow::Finished => return,
        }
    }
}

async fn answer_stub_request(
    state: &StubDaemonState,
    mut protocol_writer: StubConnectionWriter,
    daemon_request: DaemonRequest,
) -> ConnectionFlow {
    match daemon_request {
        DaemonRequest::Handshake => {
            answer_handshake(&mut protocol_writer).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::Status => {
            answer_status(state, &mut protocol_writer).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::ModelsList => {
            answer_models_list(&state.config, &mut protocol_writer).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::Catalog => {
            answer_catalog(state, &mut protocol_writer).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::DownloadStart { model_id } => {
            answer_download_start(state, &mut protocol_writer, model_id).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::DownloadStatus => {
            answer_download_status(state, &mut protocol_writer).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        model_control_request @ (DaemonRequest::ModelUnload
        | DaemonRequest::DefaultModelSet { .. }) => {
            answer_model_control(state, &mut protocol_writer, model_control_request).await;
            ConnectionFlow::KeepServing(protocol_writer)
        }
        DaemonRequest::ChatGenerate {
            messages,
            settings,
            schema_json,
            ..
        } => {
            answer_chat_generate(
                &state.config,
                protocol_writer,
                ChatRequestParts {
                    messages,
                    settings,
                    schema_json,
                },
            )
            .await;
            ConnectionFlow::Finished
        }
        DaemonRequest::TrainDecisionModel { .. } => {
            send_frames_and_close(protocol_writer, vec![training_refusal()]).await;
            ConnectionFlow::Finished
        }
        helper_model_request @ (DaemonRequest::EmbedGenerate { .. }
        | DaemonRequest::DecideGenerate { .. }
        | DaemonRequest::SpeakGenerate(_)) => {
            let helper_model_frames = scripted_helper_model_frames(
                &state.config,
                state.resident_model_id.clone(),
                helper_model_request,
            )
            .await;
            send_frames_and_close(protocol_writer, helper_model_frames).await;
            ConnectionFlow::Finished
        }
        DaemonRequest::ImageGenerate(image_request) => {
            answer_image_generate(&state.config, protocol_writer, image_request).await;
            ConnectionFlow::Finished
        }
    }
}

async fn answer_handshake(protocol_writer: &mut StubConnectionWriter) {
    protocol_writer
        .send_daemon_response(&DaemonResponse::HandshakeAccepted {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            application_name: DAEMON_APPLICATION_NAME.to_owned(),
        })
        .await
        .expect("the stub handshake should transmit");
}

async fn answer_status(state: &StubDaemonState, protocol_writer: &mut StubConnectionWriter) {
    let config = &state.config;
    wait_out_hung_status_request(&config.hung_status_requests, &state.status_request_count).await;
    let current_default_model_id = state
        .default_model_id
        .lock()
        .expect("the stub default lock should stay live")
        .clone();
    let model_load_in_flight =
        next_scripted_load_report(&config.model_load_reports, &state.model_load_report_index);
    protocol_writer
        .send_daemon_response(&DaemonResponse::Status {
            worker_status: config.worker_status,
            ready_model_id: state.resident_model_id.clone(),
            default_model_id: current_default_model_id,
            model_load_in_flight,
            request_progress_count: next_scripted_progress_count(
                &config.request_progress_counts,
                &state.request_progress_count_index,
            ),
            neural_engine_prefill: config.neural_engine_prefill.clone(),
            draft_model: config.draft_model.clone(),
            last_reply_drafting: None,
        })
        .await
        .expect("the stub status should transmit");
}

async fn answer_models_list(config: &StubDaemonConfig, protocol_writer: &mut StubConnectionWriter) {
    let listed_models: Vec<DaemonListedModel> = config
        .installed_models
        .iter()
        .map(daemon_listed_model)
        .collect();
    protocol_writer
        .send_daemon_response(&DaemonResponse::ModelsList {
            models: listed_models,
        })
        .await
        .expect("the stub model list should transmit");
}

fn daemon_listed_model(installed_model: &StubInstalledModel) -> DaemonListedModel {
    DaemonListedModel {
        model_id: installed_model.model_id.clone(),
        family: "stub".to_owned(),
        context_window: installed_model.context_window,
        supports_embeddings: installed_model.supports_embeddings,
        supports_decisions: installed_model.supports_decisions,
        supports_speech: installed_model.supports_speech,
        supports_image_generation: installed_model.supports_image_generation,
        is_resident: installed_model.is_resident,
        size_bytes: 1_000_000_000,
    }
}

async fn answer_catalog(state: &StubDaemonState, protocol_writer: &mut StubConnectionWriter) {
    let after_download_active = *state
        .download_started
        .lock()
        .expect("the stub download flag should stay live");
    let catalog_entries = if after_download_active {
        state
            .catalog_entries_after_download
            .clone()
            .expect("a catalog flip must be scripted when a download starts")
    } else {
        state.config.catalog_entries.clone()
    };
    let entries: Vec<DaemonCatalogEntry> = catalog_entries
        .into_iter()
        .map(daemon_catalog_entry)
        .collect();
    protocol_writer
        .send_daemon_response(&DaemonResponse::Catalog { entries })
        .await
        .expect("the stub catalog should transmit");
}

fn daemon_catalog_entry(catalog_entry: StubCatalogEntry) -> DaemonCatalogEntry {
    let model_id = config::leaf_model_id(&catalog_entry.huggingface_id).to_owned();
    DaemonCatalogEntry {
        huggingface_id: catalog_entry.huggingface_id,
        model_id,
        family: "stub".to_owned(),
        // ~2 GB keeps rendered sizes at two digits so
        // list-output assertions stay readable.
        approximate_size_bytes: 2_000_000_000,
        ready_on_this_mac: catalog_entry.ready_on_this_mac,
        requestable_model_id: catalog_entry.requestable_model_id,
        download_state: catalog_entry.download_state,
        context_window: catalog_entry.context_window,
        supports_reasoning: false,
        supports_vision: false,
        supports_tool_calls: false,
        supports_image_generation: catalog_entry.supports_image_generation,
        supports_embeddings: catalog_entry.supports_embeddings,
        supports_decisions: catalog_entry.supports_decisions,
        supports_speech: catalog_entry.supports_speech,
    }
}

async fn answer_download_start(
    state: &StubDaemonState,
    protocol_writer: &mut StubConnectionWriter,
    model_id: String,
) {
    *state
        .download_started
        .lock()
        .expect("the stub download flag should stay live") = true;
    protocol_writer
        .send_daemon_response(&DaemonResponse::DownloadStarted {
            huggingface_id: model_id,
        })
        .await
        .expect("the stub download start should transmit");
}

async fn answer_download_status(
    state: &StubDaemonState,
    protocol_writer: &mut StubConnectionWriter,
) {
    // The guard must drop before the send await so the task stays Send.
    let download_status = {
        let mut job_index = state
            .download_job_index
            .lock()
            .expect("the stub job index should stay live");
        *job_index += 1;
        scripted_download_status(&state.config, *job_index - 1)
    };
    protocol_writer
        .send_daemon_response(&download_status)
        .await
        .expect("the stub download status should transmit");
}

async fn answer_model_control(
    state: &StubDaemonState,
    protocol_writer: &mut StubConnectionWriter,
    model_control_request: DaemonRequest,
) {
    let control_answer = stub_model_control_answer(
        model_control_request,
        &state.config,
        &state.default_model_id,
        state.resident_model_id.as_ref(),
    );
    protocol_writer
        .send_daemon_response(&control_answer)
        .await
        .expect("the stub model control answer should transmit");
}

/// The parts of a `ChatGenerate` request the stub records and replays.
struct ChatRequestParts {
    messages: Vec<ChatMessage>,
    settings: ChatGenerationSettings,
    schema_json: Option<String>,
}

async fn answer_chat_generate(
    config: &StubDaemonConfig,
    mut protocol_writer: StubConnectionWriter,
    chat_request: ChatRequestParts,
) {
    record_chat_request(config, &chat_request);
    tokio::time::sleep(config.chat_first_frame_delay).await;
    for chat_fragment in config.chat_fragments.iter() {
        protocol_writer
            .send_daemon_response(&DaemonResponse::ChatGenerationText {
                text: chat_fragment.clone(),
            })
            .await
            .expect("the stub fragment should transmit");
    }
    protocol_writer
        .send_daemon_response(&DaemonResponse::ChatGenerationCompleted {
            prompt_token_count: 7,
            generated_token_count: 2,
            reasoning_token_count: 0,
            astro_ssd_prompt_cache_restored_token_count: 0,
            reason: ipc_protocol::ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
        .expect("the stub completion frame should transmit");
    let _ = protocol_writer.close().await;
}

/// Records the messages and settings the CLI sent so a journey test can
/// assert they crossed the boundary.
fn record_chat_request(config: &StubDaemonConfig, chat_request: &ChatRequestParts) {
    let ChatRequestParts {
        messages,
        settings,
        schema_json,
    } = chat_request;
    if let Ok(mut guard) = config.messages_capture.lock() {
        *guard = Some(messages.clone());
    }
    if let Ok(mut guard) = config.settings_capture.lock() {
        *guard = Some(settings.clone());
    }
    if let Ok(mut guard) = config.schema_capture.lock() {
        *guard = schema_json.clone();
    }
    // The user's images may sit after a system message, so find
    // the first user message rather than assuming it is first.
    if let Some(user_images) = messages.iter().find_map(|message| match message {
        ChatMessage::User { images, .. } => Some(images.clone()),
        _ => None,
    }) && let Ok(mut guard) = config.image_capture.lock()
    {
        *guard = Some(user_images);
    }
}

async fn answer_image_generate(
    config: &StubDaemonConfig,
    mut protocol_writer: StubConnectionWriter,
    image_request: ipc_protocol::ImageGenerateRequest,
) {
    *config
        .image_request_capture
        .lock()
        .expect("the stub image capture should stay live") =
        Some(DaemonRequest::ImageGenerate(image_request.clone()));
    let Some(image_response) = scripted_image_response(&config.image_outcome, image_request).await
    else {
        return;
    };
    protocol_writer
        .send_daemon_response(&image_response)
        .await
        .expect("the stub image frame should transmit");
    let _ = protocol_writer.close().await;
}
