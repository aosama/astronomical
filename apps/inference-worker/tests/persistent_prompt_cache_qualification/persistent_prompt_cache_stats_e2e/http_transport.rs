use super::*;

pub(super) fn log_cache_directory_contents(cache_directory: &std::path::Path, context: &str) {
    let kv_blocks_directory = cache_directory.join("kv_blocks");
    let recurrent_snapshots_directory = cache_directory.join("recurrent_snapshots");
    let kv_block_files = std::fs::read_dir(&kv_blocks_directory)
        .map(|directory_entries| {
            directory_entries
                .filter_map(Result::ok)
                .map(|directory_entry| directory_entry.file_name().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let recurrent_snapshot_files = std::fs::read_dir(&recurrent_snapshots_directory)
        .map(|directory_entries| {
            directory_entries
                .filter_map(Result::ok)
                .map(|directory_entry| directory_entry.file_name().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    eprintln!(
        "{context}: kv_blocks={} ({} files: {:?}), recurrent_snapshots={} ({} files: {:?})",
        kv_blocks_directory.display(),
        kv_block_files.len(),
        kv_block_files,
        recurrent_snapshots_directory.display(),
        recurrent_snapshot_files.len(),
        recurrent_snapshot_files
    );
}

pub(super) async fn get_endpoint(server_address: SocketAddr, endpoint_path: &str) -> String {
    send_http_request(
        server_address,
        format!(
            "GET {endpoint_path} HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n"
        ),
    )
    .await
}

pub(super) async fn get_cache_stats_json(server_address: SocketAddr) -> serde_json::Value {
    let cache_stats_response = get_endpoint(server_address, "/v1/cache/stats").await;
    assert!(
        cache_stats_response.starts_with("HTTP/1.1 200 OK"),
        "the cache stats endpoint should return 200 OK, got: {cache_stats_response}"
    );
    let response_body = cache_stats_response
        .split("\r\n\r\n")
        .nth(1)
        .expect("the cache stats response should have a body");
    serde_json::from_str(response_body).expect("the cache stats response body should be valid JSON")
}

async fn send_http_request(server_address: SocketAddr, request_text: String) -> String {
    let mut server_connection = TcpStream::connect(server_address)
        .await
        .expect("the E2E server should accept a local connection");
    server_connection
        .write_all(request_text.as_bytes())
        .await
        .expect("the E2E HTTP request should be written");
    let mut response_text = String::new();
    server_connection
        .read_to_string(&mut response_text)
        .await
        .expect("the bounded E2E HTTP response should be readable");
    response_text
}
