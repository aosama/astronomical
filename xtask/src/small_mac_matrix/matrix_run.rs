//! The network edge of the matrix: `curl` calls to the Development instance,
//! with progress and time left on standard error.

use std::process::Command;
use std::time::Instant;

use super::matrix_cells::{
    CellOutcome, MatrixCell, MatrixOptions, parse_chat_outcome, simulated_memory_limit_gb,
};

/// One cell loads a model and writes a few tokens; large models streaming
/// from the SSD need minutes, so the bound is generous but finite.
const CELL_TIMEOUT_SECONDS: &str = "600";

/// A fixed line of Shakespeare, the repository's standard test input.
const CHAT_PROMPT: &str = "Shall I compare thee to a summer's day?";

pub fn run_matrix(matrix_options: &MatrixOptions) -> Result<Vec<MatrixCell>, String> {
    let status = instance_status(&matrix_options.base_url)?;
    if status["application"]["channel"] != "development" {
        return Err("small-mac-matrix only runs against the Development instance".to_owned());
    }
    let original_limit_gb = status["configured_maximum_mlx_memory_gb"].clone();
    let model_ids = match matrix_options.model_ids.is_empty() {
        true => listed_model_ids(&matrix_options.base_url)?,
        false => matrix_options.model_ids.clone(),
    };
    let total_cell_count = model_ids.len() * matrix_options.mac_memory_gb.len();
    let started_at = Instant::now();
    let mut matrix_cells = Vec::with_capacity(total_cell_count);
    for model_id in &model_ids {
        for mac_memory_gb in &matrix_options.mac_memory_gb {
            let matrix_cell = run_cell(&matrix_options.base_url, model_id, *mac_memory_gb);
            matrix_cells.push(matrix_cell);
            report_progress(
                matrix_cells.len(),
                total_cell_count,
                started_at.elapsed().as_secs_f64(),
            );
        }
    }
    // Put the instance back as it was found: nothing loaded, its own limit.
    post(
        &matrix_options.base_url,
        "/v1/control/unload-model",
        "POST",
        None,
    );
    let restore_body =
        serde_json::json!({ "maximum_mlx_memory_gb": original_limit_gb }).to_string();
    post(
        &matrix_options.base_url,
        "/v1/config/maximum-mlx-memory",
        "PUT",
        Some(&restore_body),
    );
    Ok(matrix_cells)
}

fn run_cell(base_url: &str, model_id: &str, mac_memory_gb: u64) -> MatrixCell {
    let memory_limit_gb = simulated_memory_limit_gb(mac_memory_gb);
    post(base_url, "/v1/control/unload-model", "POST", None);
    let limit_body = serde_json::json!({ "maximum_mlx_memory_gb": memory_limit_gb }).to_string();
    let limit_answer = post(
        base_url,
        "/v1/config/maximum-mlx-memory",
        "PUT",
        Some(&limit_body),
    );
    let outcome = match limit_answer {
        Some((http_status, _)) if (200..300).contains(&http_status) => {
            let chat_body = serde_json::json!({
                "model": model_id,
                "messages": [{ "role": "user", "content": CHAT_PROMPT }],
                "max_tokens": 8,
            })
            .to_string();
            let asked_at = Instant::now();
            let chat_answer = post(base_url, "/v1/chat/completions", "POST", Some(&chat_body));
            let chat_answer_text = chat_answer
                .map(|(_, answer_text)| answer_text)
                .unwrap_or_default();
            parse_chat_outcome(&chat_answer_text, asked_at.elapsed().as_secs_f64())
        }
        _ => CellOutcome::LimitRefused,
    };
    MatrixCell {
        model_id: model_id.to_owned(),
        mac_memory_gb,
        memory_limit_gb,
        outcome,
    }
}

fn report_progress(completed_cell_count: usize, total_cell_count: usize, elapsed_seconds: f64) {
    let percent_complete = completed_cell_count * 100 / total_cell_count.max(1);
    let remaining_seconds = elapsed_seconds / completed_cell_count.max(1) as f64
        * total_cell_count.saturating_sub(completed_cell_count) as f64;
    eprintln!(
        "[small-mac-matrix] {completed_cell_count}/{total_cell_count} ({percent_complete}%) elapsed {elapsed_seconds:.0}s, about {remaining_seconds:.0}s left"
    );
}

fn instance_status(base_url: &str) -> Result<serde_json::Value, String> {
    let (_, status_text) = post(base_url, "/v1/status", "GET", None)
        .ok_or("the Development instance is not answering; start it with `cargo xtask app dev --background`")?;
    serde_json::from_str(&status_text).map_err(|_| "the instance's status was not JSON".to_owned())
}

fn listed_model_ids(base_url: &str) -> Result<Vec<String>, String> {
    let (_, models_text) =
        post(base_url, "/v1/models", "GET", None).ok_or("the model list did not answer")?;
    let models: serde_json::Value = serde_json::from_str(&models_text).unwrap_or_default();
    let mut model_ids = Vec::new();
    for model in models["data"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        if let Some(model_id) = model["id"].as_str() {
            model_ids.push(model_id.to_owned());
        }
    }
    Ok(model_ids)
}

/// One request; returns the HTTP status and body, or `None` when the
/// instance could not be reached.
fn post(
    base_url: &str,
    path: &str,
    method: &str,
    json_body: Option<&str>,
) -> Option<(u16, String)> {
    let mut curl = Command::new("curl");
    curl.args([
        "--silent",
        "--max-time",
        CELL_TIMEOUT_SECONDS,
        "--request",
        method,
    ]);
    curl.args(["--write-out", "\n%{http_code}"]);
    if let Some(json_body) = json_body {
        curl.args([
            "--header",
            "Content-Type: application/json",
            "--data",
            json_body,
        ]);
    }
    let curl_answer = curl.arg(format!("{base_url}{path}")).output().ok()?;
    let answer_text = String::from_utf8_lossy(&curl_answer.stdout).into_owned();
    let (body_text, status_text) = answer_text.rsplit_once('\n')?;
    let http_status: u16 = status_text.trim().parse().ok()?;
    (http_status != 0).then(|| (http_status, body_text.to_owned()))
}
