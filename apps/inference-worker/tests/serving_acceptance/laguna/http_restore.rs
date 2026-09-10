//! Repeated public HTTP generation restores an exact Laguna prompt prefix.

use std::fs;
use std::time::Duration;

use serde_json::{Value, json};
use std::net::SocketAddr;
use tokio::time::timeout;

use super::http::assert_laguna_is_advertised;
use super::validate::{
    bounded_romeo_and_juliet_source, laguna_xs_public_model_id, resolve_reference_model_directory,
};
use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, get_endpoint, launch_serving_rest_server_for_model,
    post_chat_completion, stop_serving_rest_server,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const PROMPT_CACHE_BLOCK_TOKEN_COUNT: u32 = 256;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "serves reference Laguna XS twice and proves public SSD restore"]
async fn should_restore_a_repeated_romeo_request_through_http_and_remain_ready() {
    timeout(JOURNEY_TIMEOUT, run_repeated_http_restore())
        .await
        .expect("the Laguna HTTP restore journey must finish within 115 seconds");
}

async fn run_repeated_http_restore() {
    let model_directory = resolve_reference_model_directory();
    let public_model_id = laguna_xs_public_model_id();
    let isolated_development_home =
        tempfile::tempdir().expect("an isolated Laguna cache home should be created");
    write_cache_enabled_config(
        isolated_development_home.path(),
        &public_model_id,
        &model_directory,
    );
    let rest_server = launch_serving_rest_server_for_model(
        public_model_id,
        model_directory,
        Some(isolated_development_home.path()),
        None,
    )
    .await;
    let server_address = rest_server.server_address;
    assert_laguna_is_advertised(server_address, public_model_id).await;
    let source_excerpt = bounded_romeo_and_juliet_source();
    let request_body = json!({
        "model": public_model_id,
        "messages": [{
            "role": "user",
            "content": format!("Use the supplied Romeo and Juliet source. Name the two households and the tragic ending.\n\n{source_excerpt}"),
        }],
        "stream": true,
        "temperature": 1,
        "max_tokens": 8,
    })
    .to_string();

    eprintln!("[laguna-http-restore] phase=cold");
    let cold_started_at = std::time::Instant::now();
    let cold_response = post_chat_completion(server_address, request_body.clone()).await;
    assert_successful_streaming_chat_response(&cold_response);
    let cold_cache_stats = json_endpoint(server_address, "/v1/cache/stats").await;
    assert_eq!(cold_cache_stats["persistent_prompt_cache_tokens_saved"], 0);
    eprintln!(
        "[laguna-http-restore] phase=cold status=measured elapsed_seconds={:.2}",
        cold_started_at.elapsed().as_secs_f32()
    );

    eprintln!("[laguna-http-restore] phase=warm");
    let warm_started_at = std::time::Instant::now();
    let warm_response = post_chat_completion(server_address, request_body).await;
    assert_successful_streaming_chat_response(&warm_response);
    let warm_cache_stats = json_endpoint(server_address, "/v1/cache/stats").await;
    let restored_token_count = warm_cache_stats["persistent_prompt_cache_tokens_saved"]
        .as_u64()
        .unwrap_or(0);
    assert!(
        restored_token_count >= u64::from(PROMPT_CACHE_BLOCK_TOKEN_COUNT),
        "the repeated request must restore one cache block: {warm_cache_stats}"
    );
    eprintln!(
        "[laguna-http-restore] phase=warm status=measured elapsed_seconds={:.2} restored_tokens={restored_token_count}",
        warm_started_at.elapsed().as_secs_f32()
    );
    let status_document = json_endpoint(server_address, "/v1/status").await;
    assert_eq!(status_document["status"], "ready");
    stop_serving_rest_server(rest_server).await;
}

fn write_cache_enabled_config(
    isolated_home: &std::path::Path,
    model_id: &str,
    model_directory: &std::path::Path,
) {
    let configuration_directory = isolated_home.join(".astronomical-dev");
    let models_root = configuration_directory.join("models");
    // Laguna discovery derives its immutable revision from a Hugging Face
    // cache entry, and the entry's shard links are relative to the entry
    // root, so the whole entry is published into the isolated home. A single
    // directory link keeps the multi-gigabyte weights on their original
    // volume while the worker discovers them under the public leaf id.
    let reference_cache_entry_directory = model_directory
        .parent()
        .and_then(|snapshots_directory| snapshots_directory.parent())
        .expect("the resolved reference snapshot should live inside a Hugging Face cache entry");
    let cache_entry_name = reference_cache_entry_directory
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .expect("the reference cache entry should be named");
    let decoded_model_id =
        astronomical_config::decode_huggingface_cache_directory_name(cache_entry_name)
            .expect("the reference cache entry name should decode");
    assert_eq!(
        astronomical_config::leaf_model_id(&decoded_model_id),
        model_id,
        "the reference cache entry should decode to the public leaf model id"
    );
    fs::create_dir_all(&models_root).expect("the isolated models root should be created");
    std::os::unix::fs::symlink(
        reference_cache_entry_directory,
        models_root.join(cache_entry_name),
    )
    .expect("the isolated reference cache entry link should publish");
    let configuration_document = json!({
        "model_directories": [models_root],
        "max_output_tokens": 8,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": 80,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "chunking": {

            "fixed_prompt_processing_chunk_size_tokens": 8192,
            "prompt_cache_block_tokens": PROMPT_CACHE_BLOCK_TOKEN_COUNT,
            "prompt_cache_common_prefix_stride_blocks": 1
        }
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the isolated configuration should serialize"),
    )
    .expect("the isolated configuration should write");
}

async fn json_endpoint(server_address: SocketAddr, endpoint_path: &str) -> Value {
    let http_response = get_endpoint(server_address, endpoint_path).await;
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "the {endpoint_path} endpoint should return success: {http_response}"
    );
    let (_, response_body) = http_response
        .split_once("\r\n\r\n")
        .expect("the endpoint response should contain a header/body boundary");
    serde_json::from_str(response_body).unwrap_or_else(|json_error| {
        panic!("the {endpoint_path} response should contain JSON: {json_error}")
    })
}
