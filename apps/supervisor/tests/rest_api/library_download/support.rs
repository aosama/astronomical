//! Scripted Hub transport for Library REST journeys.

use std::{collections::VecDeque, sync::Mutex, time::Duration};

use astronomical_supervisor::{
    HubHttpRequest, HubHttpResponse, HubPayloadByteStream, HubPayloadFuture, HubPayloadRequest,
    HubPayloadResponse, HubPayloadTransport, HubTransport, HubTransportError, HubTransportFuture,
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};

use super::{
    MODEL_CONFIG, MODEL_INDEX, MODEL_WEIGHTS, REPOSITORY_ID, REVISION, TOKENIZER_CONFIG,
    git_blob_sha1_hex, payload_for_request,
};

pub(super) struct ScriptedHub {
    metadata_responses: Mutex<VecDeque<HubHttpResponse>>,
    payload_delay: Duration,
    corrupt_model_config: bool,
    publishes_payload_in_two_chunks: bool,
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
        }
    }

    fn with_payload_behavior(payload_delay: Duration, corrupt_model_config: bool) -> Self {
        let config_git_blob_sha1 = git_blob_sha1_hex(MODEL_CONFIG);
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
                        [serde_json::to_vec(&serde_json::json!([
                            {
                                "type": "file",
                                "size": MODEL_CONFIG.len(),
                                "path": "config.json",
                                "oid": config_git_blob_sha1
                            },
                            {
                                "type": "file",
                                "size": TOKENIZER_CONFIG.len(),
                                "path": "tokenizer.json",
                                "oid": git_blob_sha1_hex(TOKENIZER_CONFIG)
                            },
                            {
                                "type": "file",
                                "size": MODEL_WEIGHTS.len(),
                                "path": "model-00001.safetensors",
                                "oid": git_blob_sha1_hex(MODEL_WEIGHTS)
                            },
                            {
                                "type": "file",
                                "size": MODEL_INDEX.len(),
                                "path": "model.safetensors.index.json",
                                "oid": git_blob_sha1_hex(MODEL_INDEX)
                            }
                        ]))
                        .expect("tree fixture should serialize")],
                    )
                    .expect("tree response should be valid"),
                ]
                .into(),
            ),
            payload_delay,
            corrupt_model_config,
            publishes_payload_in_two_chunks: false,
        }
    }
}

impl HubTransport for ScriptedHub {
    fn execute(&self, _request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            self.metadata_responses
                .lock()
                .map_err(|_| HubTransportError::new("metadata lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected metadata request"))
        })
    }
}

impl HubPayloadTransport for ScriptedHub {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            tokio::time::sleep(self.payload_delay).await;
            let payload = payload_for_request(&request, self.corrupt_model_config)?;
            let payload_stream: HubPayloadByteStream =
                if self.publishes_payload_in_two_chunks && payload.len() > 1 {
                    let first_payload_byte = Bytes::from_static(&payload[..1]);
                    let remaining_payload_bytes = Bytes::from_static(&payload[1..]);
                    let payload_delay = self.payload_delay;
                    Box::pin(stream::iter([Ok(first_payload_byte)]).chain(stream::once(
                        async move {
                            tokio::time::sleep(payload_delay).await;
                            Ok(remaining_payload_bytes)
                        },
                    )))
                } else {
                    Box::pin(stream::iter([Ok(Bytes::from_static(payload))]))
                };
            Ok(HubPayloadResponse::new(
                200,
                None,
                Some(payload.len() as u64),
                payload_stream,
            ))
        })
    }
}
