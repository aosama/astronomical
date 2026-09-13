//! Status-polling observation and stream consumption for the SSD-paging
//! decode expert-reuse journey, split from the journey file to keep both
//! files inside the repository's source-size budget.

use async_openai::types::stream::StreamResponse;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::time::{Duration, Instant, sleep};

use crate::support::serving_rest::{JOURNEY_TIMEOUT, get_json_endpoint};

use super::super::ssd_paging_decode_expert_reuse_journey::STATUS_LOG_INTERVAL;
use super::support::{log_status_progress, record_expert_payload_increase};

pub(super) struct ProgressiveExpertMemoryEvidence {
    pub(super) retained_expert_payload_bytes: Vec<u64>,
    pub(super) generation_expert_payload_bytes: Vec<u64>,
    pub(super) final_status: Value,
}

pub(super) async fn observe_ssd_paging_decode_expert_reuse(
    server_address: std::net::SocketAddr,
) -> ProgressiveExpertMemoryEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_prompt_processing = false;
    let mut retained_expert_payload_bytes = Vec::new();
    let mut generation_expert_payload_bytes = Vec::new();
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_status_progress(&status_document);
            last_status_log_at = Instant::now();
        }
        if status_document["activity"] == "prompt_processing" {
            observed_prompt_processing = true;
            record_expert_payload_increase(&status_document, &mut retained_expert_payload_bytes);
        }
        if observed_prompt_processing && status_document["activity"] == "generating" {
            record_expert_payload_increase(&status_document, &mut generation_expert_payload_bytes);
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_prompt_processing
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return ProgressiveExpertMemoryEvidence {
                retained_expert_payload_bytes,
                generation_expert_payload_bytes,
                final_status: status_document,
            };
        }
        assert!(Instant::now() < deadline);
        sleep(Duration::from_millis(100)).await;
    }
}

pub(super) struct CompletedStream {
    pub(super) model_text: String,
    pub(super) finish_reason: Option<String>,
}

pub(super) async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> CompletedStream {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    CompletedStream {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
    }
}
