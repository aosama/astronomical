//! The two requests the speed measurement makes to a running engine: its
//! status, to check it is set up for measuring, and one streamed chat reply.
//! They go through `curl`, as the small-Mac matrix does, so xtask needs no HTTP
//! dependency; the reply is read line by line as it streams so each token's
//! arrival time is its own.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use super::stream_timing::{ReplyMeasurement, StreamTiming};

/// Generous for a 10,000-token prompt on a slow Mac; a stuck reply still ends.
const REPLY_TIMEOUT_SECONDS: &str = "900";
const STATUS_TIMEOUT_SECONDS: &str = "10";
/// Temperature people chat at (AGENTS.md and the measurement notes: never
/// greedy for figures people read).
const MEASUREMENT_TEMPERATURE: f64 = 0.6;

pub fn fetch_status(base_url: &str) -> Result<serde_json::Value, String> {
    let status_output = Command::new("curl")
        .args(["--silent", "--fail", "--max-time", STATUS_TIMEOUT_SECONDS])
        .arg(format!("{base_url}/v1/status"))
        .output()
        .map_err(|spawn_error| format!("could not run curl: {spawn_error}"))?;
    if !status_output.status.success() {
        return Err(format!(
            "the engine at {base_url} did not answer its status; start it first"
        ));
    }
    serde_json::from_slice(&status_output.stdout)
        .map_err(|parse_error| format!("the engine's status was not JSON: {parse_error}"))
}

/// Sends one chat request and times its streamed reply.
pub fn stream_chat_reply(
    base_url: &str,
    model_id: &str,
    prompt_text: &str,
    output_tokens: u32,
) -> Result<ReplyMeasurement, String> {
    let request_body = serde_json::json!({
        "model": model_id,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": MEASUREMENT_TEMPERATURE,
        "max_tokens": output_tokens,
        "messages": [{"role": "user", "content": prompt_text}],
    })
    .to_string();
    let request_started_at = Instant::now();
    let mut curl_process = Command::new("curl")
        .args([
            "--silent",
            "--fail",
            "--no-buffer",
            "--max-time",
            REPLY_TIMEOUT_SECONDS,
        ])
        .args([
            "--header",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ])
        .arg(format!("{base_url}/v1/chat/completions"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|spawn_error| format!("could not run curl: {spawn_error}"))?;
    // The prompt goes on standard input: a 45,000-character prompt is too
    // long to pass comfortably as an argument.
    curl_process
        .stdin
        .take()
        .ok_or("curl gave no input pipe")?
        .write_all(request_body.as_bytes())
        .map_err(|write_error| format!("could not send the request: {write_error}"))?;
    let reply_stream = curl_process
        .stdout
        .take()
        .ok_or("curl gave no output pipe")?;
    let mut stream_timing = StreamTiming::default();
    for line_text in BufReader::new(reply_stream).lines().map_while(Result::ok) {
        stream_timing.record_line(&line_text, request_started_at.elapsed());
    }
    let exit_status = curl_process
        .wait()
        .map_err(|wait_error| format!("curl did not finish: {wait_error}"))?;
    if !exit_status.success() {
        return Err(format!(
            "the reply from {model_id} failed (curl exit {exit_status})"
        ));
    }
    Ok(stream_timing.measurement())
}
