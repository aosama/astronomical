use super::http_transport::get_endpoint;
use super::*;

/// Sends a streaming chat completion and prints live progress as SSE chunks
/// arrive, while concurrently polling `/v1/status` for prefill/generation
/// phase detail.
pub(super) async fn post_chat_completion_with_live_progress(
    server_address: SocketAddr,
    user_message: &str,
    maximum_output_tokens: u16,
    log_prefix: &str,
    phase_label: &str,
) -> String {
    let serialized_user_message =
        serde_json::to_string(user_message).expect("the user message should serialize into JSON");
    let request_body = format!(
        r#"{{"model":"{MODEL_ID}","messages":[{{"role":"user","content":{serialized_user_message}}}],"stream":true,"temperature":1,"max_tokens":{maximum_output_tokens}}}"#
    );
    let request_text = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request_body}",
        request_body.len()
    );

    let mut server_connection = TcpStream::connect(server_address)
        .await
        .expect("the E2E server should accept a local connection");
    server_connection
        .write_all(request_text.as_bytes())
        .await
        .expect("the E2E HTTP request should be written");

    // Spawn a background task that polls /v1/status every 2 seconds to report
    // prefill/generation progress while the streaming response is being read.
    let status_poll_task = tokio::spawn({
        let log_prefix = log_prefix.to_owned();
        let phase_label = phase_label.to_owned();
        async move {
            let mut status_interval = interval(Duration::from_millis(STATUS_POLL_INTERVAL_MILLIS));
            status_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
            status_interval.tick().await; // skip the immediate first tick
            loop {
                status_interval.tick().await;
                let status_response = get_endpoint(server_address, "/v1/status").await;
                let status_body = status_response.split("\r\n\r\n").nth(1).unwrap_or("");
                eprintln!("{log_prefix} {phase_label} status-poll: {status_body}");
            }
        }
    });

    // Read the response in chunks and print each SSE data line as it arrives.
    let mut response_buffer = Vec::with_capacity(8 * 1024);
    let mut read_buffer = [0u8; 4 * 1024];
    let mut sse_chunk_count = 0u32;
    loop {
        match server_connection.read(&mut read_buffer).await {
            Ok(0) => break,
            Ok(bytes_read) => {
                response_buffer.extend_from_slice(&read_buffer[..bytes_read]);
                // Print any complete SSE data: lines found in the newly arrived bytes.
                let new_text = String::from_utf8_lossy(&read_buffer[..bytes_read]);
                for line in new_text.lines() {
                    if line.starts_with("data: ") {
                        sse_chunk_count += 1;
                        let preview = if line.len() > 120 {
                            format!("{}...", &line[..120])
                        } else {
                            line.to_owned()
                        };
                        eprintln!(
                            "{log_prefix} {phase_label} SSE chunk #{sse_chunk_count}: {preview}"
                        );
                    }
                }
            }
            Err(read_error) => {
                status_poll_task.abort();
                panic!("the E2E HTTP response read failed: {read_error}");
            }
        }
    }
    status_poll_task.abort();

    String::from_utf8(response_buffer).expect("the E2E HTTP response should be valid UTF-8")
}

pub(super) async fn wait_until_ready(server_address: SocketAddr, log_prefix: &str) {
    let readiness_started_at = Instant::now();
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let readiness_response = get_endpoint(server_address, "/ready").await;
        if readiness_response.starts_with("HTTP/1.1 200 OK") {
            eprintln!(
                "{log_prefix} model worker ready after {readiness_attempt} attempts in {:.1}s",
                readiness_started_at.elapsed().as_secs_f64()
            );
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "{log_prefix} loading attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("{log_prefix} the model-artifact worker did not become ready before the E2E deadline");
}
