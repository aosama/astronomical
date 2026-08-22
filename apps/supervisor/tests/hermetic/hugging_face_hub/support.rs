//! Scripted Hub transport fixtures isolate protocol setup from manifest acceptance assertions.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use astronomical_supervisor::{
    HubHttpRequest, HubHttpResponse, HubTransport, HubTransportError, HubTransportFuture,
    HuggingFaceHub,
};

use super::{REPOSITORY_ID, REVISION};

pub(super) fn announce(contract: &str) {
    eprintln!("[hugging-face-hub] checking {contract}");
}

pub(super) fn hub_with_tree(tree_document: serde_json::Value) -> HuggingFaceHub {
    HuggingFaceHub::new(scripted_transport([
        valid_metadata_exchange(),
        ScriptedExchange::json(tree_url(), 200, tree_document),
    ]))
}

pub(super) fn valid_metadata_exchange() -> ScriptedExchange {
    ScriptedExchange::json(
        metadata_url(),
        200,
        serde_json::json!({
            "id": REPOSITORY_ID,
            "sha": REVISION,
            "private": false,
            "gated": false
        }),
    )
}

pub(super) fn metadata_url() -> String {
    format!("https://huggingface.co/api/models/{REPOSITORY_ID}/revision/{REVISION}")
}

pub(super) fn tree_url() -> String {
    format!("https://huggingface.co/api/models/{REPOSITORY_ID}/tree/{REVISION}?recursive=true")
}

pub(super) fn scripted_transport<const EXCHANGE_COUNT: usize>(
    exchanges: [ScriptedExchange; EXCHANGE_COUNT],
) -> Arc<ScriptedTransport> {
    Arc::new(ScriptedTransport {
        exchanges: Mutex::new(exchanges.into()),
    })
}

pub(super) struct ScriptedTransport {
    exchanges: Mutex<VecDeque<ScriptedExchange>>,
}

impl ScriptedTransport {
    pub(super) fn remaining_exchange_count(&self) -> usize {
        self.exchanges
            .lock()
            .expect("the scripted transport lock should remain available")
            .len()
    }
}

impl HubTransport for ScriptedTransport {
    fn execute(&self, request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            let exchange = self
                .exchanges
                .lock()
                .map_err(|_| HubTransportError::new("scripted transport lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected Hub request"))?;
            if request.url() != exchange.expected_url {
                return Err(HubTransportError::new(format!(
                    "expected URL {}, received {}",
                    exchange.expected_url,
                    request.url()
                )));
            }
            if request.method().as_str() != "GET"
                || request.metadata_timeout() != Duration::from_secs(30)
                || request.headers()
                    != [
                        ("Accept", "application/json"),
                        ("Accept-Encoding", "identity"),
                        (
                            "User-Agent",
                            concat!("Astronomical/", env!("CARGO_PKG_VERSION")),
                        ),
                    ]
            {
                return Err(HubTransportError::new("unexpected Hub request contract"));
            }
            Ok(exchange.response)
        })
    }
}

pub(super) struct ScriptedExchange {
    expected_url: String,
    response: HubHttpResponse,
}

impl ScriptedExchange {
    pub(super) fn empty(expected_url: String, status: u16) -> Self {
        Self {
            expected_url,
            response: HubHttpResponse::try_new(status, [], [])
                .expect("the empty scripted response should be bounded"),
        }
    }

    pub(super) fn json(expected_url: String, status: u16, body: serde_json::Value) -> Self {
        Self::json_with_headers(expected_url, status, [], body)
    }

    pub(super) fn json_with_headers<const HEADER_COUNT: usize>(
        expected_url: String,
        status: u16,
        headers: [(&str, String); HEADER_COUNT],
        body: serde_json::Value,
    ) -> Self {
        let body_bytes = serde_json::to_vec(&body).expect("the scripted body should serialize");
        let split_at = body_bytes.len() / 2;
        let body_chunks = [
            body_bytes[..split_at].to_vec(),
            body_bytes[split_at..].to_vec(),
        ];
        Self {
            expected_url,
            response: HubHttpResponse::try_new(
                status,
                headers.map(|(name, value)| (name.to_owned(), value)),
                body_chunks,
            )
            .expect("the scripted JSON response should be bounded"),
        }
    }
}
