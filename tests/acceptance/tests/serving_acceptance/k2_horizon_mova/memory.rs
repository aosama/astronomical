//! The menu must show measured MLX owners after a K2 Horizon MoVA reply.

use serde_json::json;
use tokio::time::timeout;

use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, post_chat_completion,
};

use super::support::{
    JOURNEY_TIMEOUT, assert_k2_is_advertised, households_prompt, launch_k2_rest_server,
    public_model_id, status_document, stop_k2_rest_server,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "reports MLX memory owners after serving K2 Horizon MoVA"]
async fn should_report_mlx_memory_owners_after_k2_horizon_mova_chat() {
    timeout(JOURNEY_TIMEOUT, run_memory_journey())
        .await
        .expect("the K2 Horizon MoVA memory journey must finish within 115 seconds");
}

async fn run_memory_journey() {
    let (_isolated_development_home, rest_server) = launch_k2_rest_server().await;
    let server_address = rest_server.server_address;
    assert_k2_is_advertised(server_address).await;
    let model_id = public_model_id();
    eprintln!("[k2-horizon-mova] phase=memory-chat model={model_id}");
    let chat_response = post_chat_completion(
        server_address,
        json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": households_prompt(),
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 32,
        })
        .to_string(),
    )
    .await;
    assert_successful_streaming_chat_response(&chat_response);
    let status = status_document(server_address).await;
    let snapshot = status
        .get("mlx_memory_snapshot")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let active_memory_bytes = snapshot["active_memory_bytes"].as_u64().unwrap_or(0);
    let expert_payload_bytes = snapshot["expert_payload_bytes"].as_u64().unwrap_or(0);
    let model_core_payload_bytes = snapshot["model_core_payload_bytes"].as_u64().unwrap_or(0);
    eprintln!(
        "[k2-horizon-mova] mlx_active_bytes={active_memory_bytes} expert_payload_bytes={expert_payload_bytes} model_core_payload_bytes={model_core_payload_bytes} expert_memory_mode={:?}",
        status.get("expert_memory_mode")
    );
    assert!(
        active_memory_bytes > 0,
        "K2 Horizon MoVA must report measured MLX active memory, got {snapshot}"
    );
    assert!(
        expert_payload_bytes > 0 && model_core_payload_bytes > 0,
        "K2 Horizon MoVA must attribute expert and model-core owners, got {snapshot}"
    );
    assert_eq!(
        status["expert_memory_mode"], "resident",
        "fully seated K2 Horizon MoVA must report resident expert memory: {status}"
    );
    stop_k2_rest_server(rest_server).await;
}
