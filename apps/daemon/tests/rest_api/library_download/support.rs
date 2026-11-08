//! Scripted Hub transport for Library REST journeys.

use std::{collections::VecDeque, sync::Mutex, time::Duration};

use astronomical_daemon::{
    HubHttpRequest, HubHttpResponse, HubPayloadByteStream, HubPayloadFuture, HubPayloadRequest,
    HubPayloadResponse, HubPayloadTransport, HubTransport, HubTransportError, HubTransportFuture,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};

use super::{
    MODEL_CONFIG, MODEL_INDEX, MODEL_WEIGHTS, REPOSITORY_ID, REVISION, TOKENIZER_CONFIG,
    git_blob_sha1_hex, payload_for_request,
};

/// Where the empty file sits in the `with_empty_file` repository.
pub(super) const EMPTY_FILE_PATH: &str = "tools/__init__.py";

pub(super) struct ScriptedHub {
    metadata_responses: Mutex<VecDeque<HubHttpResponse>>,
    payload_delay: Duration,
    corrupt_model_config: bool,
    publishes_payload_in_two_chunks: bool,
    drops_every_payload_connection: bool,
}

impl ScriptedHub {
    pub(super) fn new() -> Self {
        Self::with_payload_behavior(Duration::ZERO, false)
    }

    pub(super) fn delayed(payload_delay: Duration) -> Self {
        Self::with_payload_behavior(payload_delay, false)
    }

    pub(super) fn checksum_mismatch() -> Self {
        Self::with_payload_behavior(Duration::ZERO, true)
    }

    /// A repository that also ships an empty file, as Hugging Face repositories
    /// with Python tooling often do (`__init__.py`); issue #1896.
    pub(super) fn with_empty_file() -> Self {
        Self::with_repository_files(Duration::ZERO, false, true)
    }

    /// The network drops every payload request, as a lost Wi-Fi connection or a provider
    /// outage would; the download ends `failed` with `download_failed`.
    pub(super) fn network_failure() -> Self {
        let mut hub = Self::new();
        hub.drops_every_payload_connection = true;
        hub
    }

    pub(super) fn progressive(payload_delay: Duration) -> Self {
        let mut hub = Self::with_payload_behavior(payload_delay, false);
        hub.publishes_payload_in_two_chunks = true;
        hub
    }

    pub(super) fn gated() -> Self {
        Self {
            metadata_responses: Mutex::new(
                [HubHttpResponse::try_new(403, [], [b"{}".to_vec()])
                    .expect("gated response should be valid")]
                .into(),
            ),
            payload_delay: Duration::ZERO,
            corrupt_model_config: false,
            publishes_payload_in_two_chunks: false,
            drops_every_payload_connection: false,
        }
    }

    fn with_payload_behavior(payload_delay: Duration, corrupt_model_config: bool) -> Self {
        Self::with_repository_files(payload_delay, corrupt_model_config, false)
    }

    fn with_repository_files(
        payload_delay: Duration,
        corrupt_model_config: bool,
        includes_empty_file: bool,
    ) -> Self {
        Self {
            metadata_responses: Mutex::new(
                [
                    HubHttpResponse::try_new(
                        200,
                        [],
                        [serde_json::to_vec(&serde_json::json!({
                            "id": REPOSITORY_ID,
                            "sha": REVISION,
                            "private": false,
                            "gated": false
                        }))
                        .expect("metadata fixture should serialize")],
                    )
                    .expect("metadata response should be valid"),
                    HubHttpResponse::try_new(
                        200,
                        [],
                        [serde_json::to_vec(&repository_tree(includes_empty_file))
                            .expect("tree fixture should serialize")],
                    )
                    .expect("tree response should be valid"),
                ]
                .into(),
            ),
            payload_delay,
            corrupt_model_config,
            publishes_payload_in_two_chunks: false,
            drops_every_payload_connection: false,
        }
    }
}

/// The repository's file listing, as the Hub's tree endpoint returns it.
fn repository_tree(includes_empty_file: bool) -> serde_json::Value {
    let mut tree_entries = vec![
        tree_entry("config.json", MODEL_CONFIG),
        tree_entry("tokenizer.json", TOKENIZER_CONFIG),
        tree_entry("model-00001.safetensors", MODEL_WEIGHTS),
        tree_entry("model.safetensors.index.json", MODEL_INDEX),
    ];
    if includes_empty_file {
        tree_entries.push(tree_entry(EMPTY_FILE_PATH, b""));
    }
    serde_json::Value::Array(tree_entries)
}

fn tree_entry(path: &str, contents: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "type": "file",
        "size": contents.len(),
        "path": path,
        "oid": git_blob_sha1_hex(contents)
    })
}

impl HubTransport for ScriptedHub {
    fn execute(&self, request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            // The preflight executable gate reads small Git files through `/raw/` origin
            // endpoints, which the real Hub routes by URL. Mirror that routing so scripted
            // journeys exercise the gate instead of failing on an exhausted response queue.
            let request_url = request.url();
            if request_url.ends_with(&format!("/raw/{REVISION}/config.json")) {
                return Ok(ok_bytes(MODEL_CONFIG));
            }
            if request_url.ends_with(&format!("/raw/{REVISION}/model.safetensors.index.json")) {
                return Ok(ok_bytes(MODEL_INDEX));
            }
            self.metadata_responses
                .lock()
                .map_err(|_| HubTransportError::new("metadata lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected metadata request"))
        })
    }
}

fn ok_bytes(body: &'static [u8]) -> HubHttpResponse {
    HubHttpResponse::try_new(200, [], [body.to_vec()])
        .expect("scripted raw file response should be valid")
}

impl ScriptedHub {
    /// The payload as one chunk, or as two separated by the payload delay so a journey can
    /// observe progress mid-file.
    fn scripted_payload_stream(&self, payload: &'static [u8]) -> HubPayloadByteStream {
        if !self.publishes_payload_in_two_chunks || payload.len() <= 1 {
            return Box::pin(stream::iter([Ok(Bytes::from_static(payload))]));
        }
        let first_payload_byte = Bytes::from_static(&payload[..1]);
        let remaining_payload_bytes = Bytes::from_static(&payload[1..]);
        let payload_delay = self.payload_delay;
        Box::pin(
            stream::iter([Ok(first_payload_byte)]).chain(stream::once(async move {
                tokio::time::sleep(payload_delay).await;
                Ok(remaining_payload_bytes)
            })),
        )
    }
}

impl HubPayloadTransport for ScriptedHub {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            tokio::time::sleep(self.payload_delay).await;
            if self.drops_every_payload_connection {
                return Err(HubTransportError::new("scripted connection reset"));
            }
            let payload = payload_for_request(&request, self.corrupt_model_config)?;
            let payload_stream = self.scripted_payload_stream(payload);
            Ok(HubPayloadResponse::new(
                200,
                None,
                Some(payload.len() as u64),
                payload_stream,
            ))
        })
    }
}
