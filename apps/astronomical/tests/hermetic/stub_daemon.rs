//! Configurable stub daemon speaking the real framed protocol over a real
//! unix socket, shared by the one-shot verb journey tests.
//!
//! It mirrors the daemon's lifecycle surface: status, installed models,
//! catalog, download start/status, default-model persistence, and scripted
//! chat or embeddings results. One connection serves a scripted exchange
//! sequence, like the real daemon.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use astronomical_ipc_protocol::{
    DAEMON_APPLICATION_NAME, DAEMON_PROTOCOL_VERSION, DaemonCatalogEntry, DaemonDownloadJob,
    DaemonListedModel, DaemonRequest, DaemonResponse, DaemonWorkerStatus, EmbeddingsFailureReason,
    ProtocolReader, ProtocolWriter,
};

/// One installed model the stub reports for `ModelsList`.
pub struct StubInstalledModel {
    pub model_id: String,
    pub context_window: Option<u32>,
    pub supports_embeddings: bool,
    pub is_resident: bool,
}

impl StubInstalledModel {
    /// A chat-capable installed model.
    pub fn chat(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: Some(32_768),
            supports_embeddings: false,
            is_resident,
        }
    }

    /// An embeddings-only installed model.
    pub fn embeddings(model_id: &str, is_resident: bool) -> Self {
        StubInstalledModel {
            model_id: model_id.to_owned(),
            context_window: None,
            supports_embeddings: true,
            is_resident,
        }
    }
}

/// One catalog entry the stub reports for `Catalog`.
#[derive(Clone)]
pub struct StubCatalogEntry {
    pub huggingface_id: String,
    pub requestable_model_id: Option<String>,
    pub ready_on_this_mac: bool,
    pub download_state: Option<String>,
    pub context_window: Option<u32>,
    pub supports_embeddings: bool,
}

impl StubCatalogEntry {
    /// A downloadable chat model.
    pub fn chat(huggingface_id: &str, requestable_model_id: &str, ready_on_this_mac: bool) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: Some(32_768),
            supports_embeddings: false,
        }
    }

    /// A downloadable embeddings-only model.
    pub fn embeddings(
        huggingface_id: &str,
        requestable_model_id: &str,
        ready_on_this_mac: bool,
    ) -> Self {
        StubCatalogEntry {
            huggingface_id: huggingface_id.to_owned(),
            requestable_model_id: Some(requestable_model_id.to_owned()),
            ready_on_this_mac,
            download_state: None,
            context_window: None,
            supports_embeddings: true,
        }
    }
}

/// Scripted result of an embeddings request.
pub enum StubEmbeddingsOutcome {
    Completed,
    ContextLengthExceeded,
}

/// Scripted behaviour of the stub daemon for one test.
pub struct StubDaemonConfig {
    pub worker_status: DaemonWorkerStatus,
    pub installed_models: Vec<StubInstalledModel>,
    pub catalog_entries: Vec<StubCatalogEntry>,
    /// Replaces `catalog_entries` for `Catalog` requests once a download has
    /// been started — how the daemon reports a just-finished download.
    pub catalog_entries_after_download: Option<Vec<StubCatalogEntry>>,
    pub default_model_id: Option<String>,
    pub chat_fragments: Vec<String>,
    pub embeddings_outcome: StubEmbeddingsOutcome,
    pub embedding_vector: Vec<f32>,
    /// FIFO of jobs for `DownloadStatus` requests; the last entry repeats
    /// once the scripted list is exhausted. `None` means no active job.
    pub download_jobs: Vec<Option<DaemonDownloadJob>>,
}

impl Default for StubDaemonConfig {
    fn default() -> Self {
        StubDaemonConfig {
            worker_status: DaemonWorkerStatus::Ready,
            installed_models: Vec::new(),
            catalog_entries: Vec::new(),
            catalog_entries_after_download: None,
            default_model_id: None,
            chat_fragments: Vec::new(),
            embeddings_outcome: StubEmbeddingsOutcome::Completed,
            embedding_vector: vec![0.25, -0.5],
            download_jobs: Vec::new(),
        }
    }
}

impl StubDaemonConfig {
    /// The model the daemon reports as resident: the first resident
    /// installed model, mirroring the real daemon's status frame.
    fn resident_model_id(&self) -> Option<String> {
        self.installed_models
            .iter()
            .find(|installed_model| installed_model.is_resident)
            .map(|installed_model| installed_model.model_id.clone())
    }
}

/// Binds the socket and serves scripted exchanges until aborted.
pub fn spawn_stub_daemon(
    socket_path: PathBuf,
    config: StubDaemonConfig,
) -> tokio::task::JoinHandle<()> {
    let resident_model_id = config.resident_model_id();
    let catalog_entries_after_download = Arc::new(config.catalog_entries_after_download.clone());
    let default_model_id = Arc::new(Mutex::new(config.default_model_id.clone()));
    let download_job_index = Arc::new(Mutex::new(0usize));
    let download_started = Arc::new(Mutex::new(false));
    // Bind before spawning so the client can never race an unbound socket.
    let std_listener = std::os::unix::net::UnixListener::bind(&socket_path)
        .expect("the stub daemon should bind the test socket");
    std_listener
        .set_nonblocking(true)
        .expect("the stub listener should accept non-blocking mode");
    let unix_listener = tokio::net::UnixListener::from_std(std_listener)
        .expect("the stub listener should register with the runtime");
    let config = Arc::new(config);
    tokio::spawn(async move {
        loop {
            let Ok((connection_stream, _peer_address)) = unix_listener.accept().await else {
                return;
            };
            let config = Arc::clone(&config);
            let resident_model_id = resident_model_id.clone();
            let catalog_entries_after_download = Arc::clone(&catalog_entries_after_download);
            let default_model_id = Arc::clone(&default_model_id);
            let download_job_index = Arc::clone(&download_job_index);
            let download_started = Arc::clone(&download_started);
            tokio::spawn(async move {
                let (read_half, write_half) = connection_stream.into_split();
                let mut protocol_reader = ProtocolReader::new(read_half);
                let mut protocol_writer = ProtocolWriter::new(write_half);
                loop {
                    let Some(daemon_request) = protocol_reader.next_daemon_request().await.expect(
                        "the stub daemon request read should not fail at the transport layer",
                    ) else {
                        return;
                    };
                    match daemon_request {
                        DaemonRequest::Handshake => {
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::HandshakeAccepted {
                                    protocol_version: DAEMON_PROTOCOL_VERSION,
                                    application_name: DAEMON_APPLICATION_NAME.to_owned(),
                                })
                                .await
                                .expect("the stub handshake should transmit");
                        }
                        DaemonRequest::Status => {
                            let current_default_model_id = default_model_id
                                .lock()
                                .expect("the stub default lock should stay live")
                                .clone();
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::Status {
                                    worker_status: config.worker_status,
                                    ready_model_id: resident_model_id.clone(),
                                    default_model_id: current_default_model_id,
                                })
                                .await
                                .expect("the stub status should transmit");
                        }
                        DaemonRequest::ModelsList => {
                            let listed_models: Vec<DaemonListedModel> = config
                                .installed_models
                                .iter()
                                .map(|installed_model| DaemonListedModel {
                                    model_id: installed_model.model_id.clone(),
                                    family: "stub".to_owned(),
                                    context_window: installed_model.context_window,
                                    supports_embeddings: installed_model.supports_embeddings,
                                    is_resident: installed_model.is_resident,
                                    size_bytes: 1_000_000_000,
                                })
                                .collect();
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::ModelsList {
                                    models: listed_models,
                                })
                                .await
                                .expect("the stub model list should transmit");
                        }
                        DaemonRequest::Catalog => {
                            let after_download_active = *download_started
                                .lock()
                                .expect("the stub download flag should stay live");
                            let catalog_entries = if after_download_active {
                                catalog_entries_after_download.as_ref().clone().expect(
                                    "a catalog flip must be scripted when a download starts",
                                )
                            } else {
                                config.catalog_entries.clone()
                            };
                            let entries: Vec<DaemonCatalogEntry> = catalog_entries
                                .into_iter()
                                .map(|catalog_entry| {
                                    let display_name = catalog_entry
                                        .requestable_model_id
                                        .clone()
                                        .unwrap_or(catalog_entry.huggingface_id.clone());
                                    DaemonCatalogEntry {
                                        huggingface_id: catalog_entry.huggingface_id,
                                        display_name,
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
                                        supports_image_generation: false,
                                        supports_embeddings: catalog_entry.supports_embeddings,
                                    }
                                })
                                .collect();
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::Catalog { entries })
                                .await
                                .expect("the stub catalog should transmit");
                        }
                        DaemonRequest::DownloadStart { model_id } => {
                            *download_started
                                .lock()
                                .expect("the stub download flag should stay live") = true;
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DownloadStarted {
                                    huggingface_id: model_id,
                                })
                                .await
                                .expect("the stub download start should transmit");
                        }
                        DaemonRequest::DownloadStatus => {
                            // The guard must drop before the send await so the task stays Send.
                            let scripted_job = {
                                let mut job_index = download_job_index
                                    .lock()
                                    .expect("the stub job index should stay live");
                                let scripted_job = if *job_index < config.download_jobs.len() {
                                    config.download_jobs[*job_index].clone()
                                } else {
                                    config.download_jobs.last().cloned().unwrap_or(None)
                                };
                                *job_index += 1;
                                scripted_job
                            };
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DownloadStatus {
                                    job: scripted_job,
                                })
                                .await
                                .expect("the stub download status should transmit");
                        }
                        DaemonRequest::DefaultModelSet { model_id } => {
                            *default_model_id
                                .lock()
                                .expect("the stub default lock should stay live") =
                                Some(model_id.clone());
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DefaultModelSet {
                                    default_model_id: model_id,
                                })
                                .await
                                .expect("the stub default set should transmit");
                        }
                        DaemonRequest::ChatGenerate { .. } => {
                            for chat_fragment in config.chat_fragments.iter() {
                                protocol_writer
                                    .send_daemon_response(&DaemonResponse::ChatGenerationText {
                                        text: chat_fragment.clone(),
                                    })
                                    .await
                                    .expect("the stub fragment should transmit");
                            }
                            protocol_writer
                                .send_daemon_response(
                                    &DaemonResponse::ChatGenerationCompleted {
                                        prompt_token_count: 7,
                                        generated_token_count: 2,
                                        reasoning_token_count: 0,
                                        cached_token_count: 0,
                                        reason:
                                            astronomical_ipc_protocol::ChatGenerationCompletionReason::EndOfSequence,
                                    },
                                )
                                .await
                                .expect("the stub completion frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                        DaemonRequest::EmbedGenerate { model, .. } => {
                            let embeddings_response = match config.embeddings_outcome {
                                StubEmbeddingsOutcome::Completed => {
                                    DaemonResponse::EmbeddingsCompleted {
                                        model: model.unwrap_or_else(|| {
                                            resident_model_id
                                                .clone()
                                                .expect("the completed stub always has a model")
                                        }),
                                        vectors: vec![config.embedding_vector.clone()],
                                        input_token_counts: vec![3],
                                    }
                                }
                                StubEmbeddingsOutcome::ContextLengthExceeded => {
                                    DaemonResponse::EmbeddingsFailed {
                                        reason: EmbeddingsFailureReason::ContextLengthExceeded {
                                            actual_total_context_tokens: 5_000,
                                            maximum_context_tokens: 2_048,
                                        },
                                    }
                                }
                            };
                            protocol_writer
                                .send_daemon_response(&embeddings_response)
                                .await
                                .expect("the stub embeddings frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                    }
                }
            });
        }
    })
}

/// One scripted download job for the stub's `DownloadStatus` sequence.
pub fn stub_download_job(
    huggingface_id: &str,
    state: &str,
    bytes_completed: u64,
    bytes_total: u64,
    error: Option<&str>,
) -> DaemonDownloadJob {
    DaemonDownloadJob {
        huggingface_id: huggingface_id.to_owned(),
        state: state.to_owned(),
        bytes_completed,
        bytes_total,
        error: error.map(str::to_owned),
    }
}
