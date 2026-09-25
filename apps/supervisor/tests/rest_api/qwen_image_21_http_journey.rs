//! Live end-to-end acceptance journey: one Qwen-Image-2.1 image over real HTTP.
//!
//! This ignored journey spawns the actual `astronomicald` daemon (which discovers and spawns the
//! real inference worker from its own directory), configures it to discover the installed
//! Qwen-Image-2.1 artifact, and serves one 256×256 request through
//! `POST /v1/images/generations`. It is the only journey that exercises the whole chain in one
//! live process tree — discovery advertisement, worker startup, model swap, the native engine,
//! and the REST response contract — so a green run here means the code actually works, not that
//! each layer works in isolation.
//!
//! The render runs at 256×256 with the REST contract's fixed four steps so the whole journey
//! fits the bounded 115-second budget on any machine that can hold the model. Setting
//! `ASTRONOMICAL_QWEN_IMAGE_21_RENDER_OUTPUT` writes the served PNG to that path so a human can
//! inspect it for coherence — pixel-statistics assertions cannot see garbling that keeps
//! neighbour pixels correlated.
//!
//! Prerequisites: the real worker binary must sit next to the daemon binary, so build both
//! before running (`cargo build -p astronomical-supervisor -p astronomical-inference-worker`).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use base64::Engine;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::time::timeout;

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const ARTIFACT_ENV: &str = "ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY";
const RENDER_OUTPUT_ENV: &str = "ASTRONOMICAL_QWEN_IMAGE_21_RENDER_OUTPUT";
const DAEMON_STARTUP_PREFIX: &str = "astronomicald listening on http://";
/// A prompt from the repo's mandated Romeo and Juliet source text.
const JOURNEY_PROMPT: &str =
    "It is the east, and Juliet is the sun. Golden dawn light over a balcony garden.";
const JOURNEY_WIDTH: u32 = 256;
const JOURNEY_HEIGHT: u32 = 256;
/// The REST image contract accepts exactly four steps and guidance 1.0.
const JOURNEY_STEPS: u32 = 4;
const JOURNEY_SEED: u64 = 20260924;
const EXPECTED_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
/// Minimum neighbour luma correlation the served image must clear to count as a picture.
const MINIMUM_NEIGHBOR_CORRELATION: f64 = 0.5;
/// Startup (worker spawn + discovery + swap + render) must stay well inside the journey bound.
const ENDPOINT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[ignore = "requires the Qwen-Image-2.1 artifact directory and built daemon+worker binaries; \
set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_serve_one_qwen_image_over_http_end_to_end() {
    timeout(JOURNEY_TIMEOUT, async {
        let model_root = qwen_image_model_root();
        let state_directory = tempfile::tempdir().expect("state directory should be created");
        let isolated_home = tempfile::tempdir().expect("isolated home should be created");
        write_instance_config(
            state_directory.path(),
            &model_root.to_string_lossy(),
        );

        let (mut daemon_process, daemon_address) =
            spawn_real_daemon(state_directory.path(), isolated_home.path()).await;

        wait_for_model_advertisement(daemon_address).await;

        let http_response =
            post_image_generation_request_with_startup_retries(daemon_address, EXPECTED_MODEL_ID)
                .await;
        let served_image = parse_served_image(&http_response);

        let decoded_png =
            image::load_from_memory(&served_image.png_bytes)
                .expect("the served base64 payload should be one decodable PNG");
        assert_eq!(
            (decoded_png.width(), decoded_png.height()),
            (JOURNEY_WIDTH, JOURNEY_HEIGHT),
            "the served image must carry the requested dimensions"
        );
        assert_eq!(served_image.mime_type, "image/png");
        assert_eq!(served_image.seed, JOURNEY_SEED);
        assert!(
            !served_image.model_revision.is_empty(),
            "the response must carry the artifact revision for reproducibility"
        );
        let neighbor_correlation = horizontal_luma_correlation(
            decoded_png.to_rgb8().as_raw(),
            JOURNEY_WIDTH as usize,
            JOURNEY_HEIGHT as usize,
        );
        assert!(
            neighbor_correlation >= MINIMUM_NEIGHBOR_CORRELATION,
            "the served image must be spatially coherent, not noise: neighbour luma \
             correlation {neighbor_correlation:.4} is below the {MINIMUM_NEIGHBOR_CORRELATION} floor"
        );

        if let Some(output_path) = std::env::var_os(RENDER_OUTPUT_ENV) {
            let output_path = PathBuf::from(output_path);
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)
                    .expect("the journey output parent should create");
            }
            std::fs::write(&output_path, &served_image.png_bytes)
                .unwrap_or_else(|error| panic!("the journey output should write: {error}"));
        }

        terminate_daemon(&daemon_process);
        let daemon_exit = daemon_process
            .wait()
            .await
            .expect("the daemon should be reaped after termination");
        assert!(
            daemon_exit.success(),
            "the daemon should shut down cleanly, exited with {daemon_exit}"
        );
    })
    .await
    .expect("the live HTTP journey should finish within its 115 s budget");
}

/// The `models--org--repo` directory containing the installed snapshot, resolved from the
/// shared artifact environment variable so no developer path is hardcoded.
fn qwen_image_model_root() -> PathBuf {
    let artifact_directory = match std::env::var_os(ARTIFACT_ENV) {
        Some(value) if !value.is_empty() => PathBuf::from(value),
        _ => panic!(
            "set {ARTIFACT_ENV} to the installed Qwen-Image-2.1 artifact root (the \
             models--mlx-community--Qwen-Image-2.1-MLX-4bit snapshots directory), not a \
             hardcoded developer path"
        ),
    };
    // The env var points at `models--.../snapshots/<sha>`; discovery accepts the `models--...`
    // entry itself and resolves the active snapshot with its decoded model id.
    let model_root = artifact_directory
        .parent()
        .and_then(Path::parent)
        .expect("the artifact root should sit at models--<org>--<repo>/snapshots/<sha>")
        .to_path_buf();
    assert!(
        model_root
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("models--")),
        "the derived model root should be one Hugging Face cache entry, found {}",
        model_root.display()
    );
    model_root
}

/// The real daemon resolves configuration from the state directory it is given.
fn write_instance_config(state_directory: &Path, model_root: &str) {
    let configuration_document = format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":["{model_root}"]}},"chunking":{{"fixed_prompt_processing_chunk_size_tokens":2048}}}}"#
    );
    std::fs::write(state_directory.join("config.json"), configuration_document)
        .expect("instance config should be written");
}

async fn spawn_real_daemon(state_directory: &Path, isolated_home: &Path) -> (Child, SocketAddr) {
    let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
        .expect("Cargo should provide the astronomicald executable path");
    assert_real_worker_binary_exists(&daemon_executable_path);
    let mut daemon_process = Command::new(daemon_executable_path)
        .args(["--instance", "development", "--state-directory"])
        .arg(state_directory)
        .env("HOME", isolated_home)
        .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("the real daemon process should start");
    let daemon_stdout = daemon_process
        .stdout
        .take()
        .expect("daemon stdout should be captured");
    let mut stdout_lines = BufReader::new(daemon_stdout).lines();
    let startup_line = timeout(Duration::from_secs(20), stdout_lines.next_line())
        .await
        .expect("the daemon should report its endpoint")
        .expect("daemon startup output should be readable")
        .expect("daemon startup output should not end");
    let daemon_address = startup_line
        .strip_prefix(DAEMON_STARTUP_PREFIX)
        .expect("the daemon should report the expected startup prefix")
        .parse()
        .expect("the daemon address should parse");
    (daemon_process, daemon_address)
}

/// The real daemon spawns its sibling `astronomical-inference-worker`; that binary must be
/// built before the journey runs, which the panic message spells out.
fn assert_real_worker_binary_exists(daemon_executable_path: &str) {
    let daemon_directory = Path::new(daemon_executable_path)
        .parent()
        .expect("the daemon executable should have a parent directory");
    let worker_executable_path = daemon_directory.join("astronomical-inference-worker");
    assert!(
        worker_executable_path.is_file(),
        "the real worker binary must exist next to the daemon: build it first with \
         `cargo build -p astronomical-inference-worker` (expected {})",
        worker_executable_path.display()
    );
}

async fn wait_for_model_advertisement(daemon_address: SocketAddr) {
    let advertisement_deadline = Instant::now() + ENDPOINT_WAIT_TIMEOUT;
    loop {
        let models_response = get_endpoint(daemon_address, "/v1/models").await;
        if models_response.contains(EXPECTED_MODEL_ID) {
            return;
        }
        assert!(
            Instant::now() < advertisement_deadline,
            "the daemon should advertise {EXPECTED_MODEL_ID} within \
             {ENDPOINT_WAIT_TIMEOUT:?}; latest /v1/models response: {models_response}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// Posts the image request, retrying only while the worker process is still completing its
/// startup handshake — the daemon reports its endpoint and discovered models before the worker
/// is ready, and a request raced against that handshake fails fast with `worker_unavailable`.
async fn post_image_generation_request_with_startup_retries(
    daemon_address: SocketAddr,
    model_id: &str,
) -> String {
    let startup_deadline = Instant::now() + ENDPOINT_WAIT_TIMEOUT;
    loop {
        let http_response = post_image_generation_request(daemon_address, model_id).await;
        if !http_response.contains("worker_unavailable") {
            return http_response;
        }
        assert!(
            Instant::now() < startup_deadline,
            "the worker should complete its startup handshake within {ENDPOINT_WAIT_TIMEOUT:?}; \
             latest image response: {}",
            bounded_response(&http_response)
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn post_image_generation_request(daemon_address: SocketAddr, model_id: &str) -> String {
    let request_body = format!(
        r#"{{"model":"{model_id}","prompt":"{JOURNEY_PROMPT}","width":{JOURNEY_WIDTH},"height":{JOURNEY_HEIGHT},"steps":{JOURNEY_STEPS},"guidance":1.0,"response_format":"b64_json","seed":{JOURNEY_SEED}}}"#
    );
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local image connection");
    daemon_connection
        .write_all(
            format!(
                "POST /v1/images/generations HTTP/1.1\r\nHost: {daemon_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                request_body.len(),
                request_body
            )
            .as_bytes(),
        )
        .await
        .expect("the image request should be written");
    let mut http_response = String::new();
    daemon_connection
        .read_to_string(&mut http_response)
        .await
        .expect("the image response should be readable");
    http_response
}

async fn get_endpoint(daemon_address: SocketAddr, endpoint_path: &str) -> String {
    let mut daemon_connection = TcpStream::connect(daemon_address)
        .await
        .expect("the daemon should accept a local connection");
    daemon_connection
        .write_all(
            format!("GET {endpoint_path} HTTP/1.1\r\nHost: {daemon_address}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .await
        .expect("the endpoint request should be written");
    let mut endpoint_response = String::new();
    daemon_connection
        .read_to_string(&mut endpoint_response)
        .await
        .expect("the endpoint response should be readable");
    endpoint_response
}

struct ServedImage {
    png_bytes: Vec<u8>,
    mime_type: String,
    model_revision: String,
    seed: u64,
}

fn parse_served_image(http_response: &str) -> ServedImage {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "the image request should succeed, got: {}",
        bounded_response(http_response)
    );
    let response_body = http_response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("the HTTP response should separate headers from the body");
    let response_document: serde_json::Value =
        serde_json::from_str(response_body).unwrap_or_else(|error| {
            panic!(
                "the image response body should be JSON: {error}; body: {}",
                bounded_response(response_body)
            )
        });
    let served_image = &response_document["data"][0];
    let b64_json = served_image["b64_json"]
        .as_str()
        .expect("the response should carry one base64 image");
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_json)
        .expect("the served image should be valid base64");
    assert!(
        png_bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
        "the served payload must carry the PNG magic header"
    );
    ServedImage {
        png_bytes,
        mime_type: served_image["mime_type"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        model_revision: served_image["model_revision"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        seed: served_image["seed"].as_u64().unwrap_or_default(),
    }
}

fn bounded_response(response: &str) -> String {
    response.chars().take(2_048).collect()
}

fn terminate_daemon(daemon_process: &Child) {
    let process_id = daemon_process.id().expect("the daemon should still run");
    let terminate_status = std::process::Command::new("kill")
        .args(["-TERM", &process_id.to_string()])
        .status()
        .expect("the test should send SIGTERM");
    assert!(terminate_status.success());
}

/// Pearson correlation between neighbouring pixels' luma — the cheapest separator between a
/// picture and noise. A real render scores well above 0.8 at these dimensions; uncorrelated
/// noise scores near 0, so the floor sits far from both.
fn horizontal_luma_correlation(rgb_bytes: &[u8], width: usize, height: usize) -> f64 {
    let mut left_pixels = Vec::with_capacity(width * height);
    let mut right_pixels = Vec::with_capacity(width * height);
    for row_index in 0..height {
        let row_start = row_index * width * 3;
        for column_index in 0..width - 1 {
            left_pixels.push(luma_of(&rgb_bytes[row_start + column_index * 3..]));
            right_pixels.push(luma_of(&rgb_bytes[row_start + (column_index + 1) * 3..]));
        }
    }
    let left_mean = left_pixels.iter().sum::<f64>() / left_pixels.len() as f64;
    let right_mean = right_pixels.iter().sum::<f64>() / right_pixels.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left_pixels.iter().zip(right_pixels.iter()) {
        let left_offset = left - left_mean;
        let right_offset = right - right_mean;
        covariance += left_offset * right_offset;
        left_variance += left_offset * left_offset;
        right_variance += right_offset * right_offset;
    }
    covariance / (left_variance * right_variance).sqrt()
}

fn luma_of(rgb_triple: &[u8]) -> f64 {
    0.299 * f64::from(rgb_triple[0])
        + 0.587 * f64::from(rgb_triple[1])
        + 0.114 * f64::from(rgb_triple[2])
}
