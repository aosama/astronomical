use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use astronomical_ipc_protocol::{ChatGenerationCommand, ExpertMemoryMode};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationPerformanceLog,
    ResolvedRuntimeConfigResolver, WorkerHandle, WorkerHealthStatus,
};
use serde_json::json;
use tokio::time::sleep;

use crate::common::exact_model_prompt::build_exact_model_prompt_content;

use super::model_prefill_benchmark_report::PrefillMeasurementAccumulator;

pub(super) const PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS: u16 = 512;
pub(super) const PREFILL_QUALIFICATION_MODEL_ID: &str =
    crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
const PREFILL_QUALIFICATION_SOURCE_DOCUMENT: &str =
    include_str!("../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");
const EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT: u8 = 70;
const PREFILL_QUALIFICATION_TRANSIENT_MLX_PEAK_ALLOWANCE_DIVISOR: u64 = 100;

pub(super) struct PreparedPrefillQualificationWorker {
    isolated_worker_home: IsolatedPrefillQualificationWorkerHome,
    pub(super) worker_executable_path: PathBuf,
    pub(super) optimizer_state_loaded_before_run: bool,
}

impl PreparedPrefillQualificationWorker {
    pub(super) fn isolated_worker_home_directory(&self) -> &Path {
        self.isolated_worker_home.directory_path()
    }

    pub(super) fn optimizer_state_file_path(&self) -> PathBuf {
        self.isolated_worker_home_directory()
            .join(".astronomical")
            .join("optimizer")
            .join("prefill-chunck-size.json")
    }
}

struct IsolatedPrefillQualificationWorkerHome {
    directory_path: PathBuf,
    _temporary_directory_owner: Option<tempfile::TempDir>,
}

impl IsolatedPrefillQualificationWorkerHome {
    fn from_environment_or_temporary() -> Self {
        if let Some(configured_directory_path) =
            std::env::var_os("ASTRONOMICAL_BENCHMARK_WORKER_HOME").map(PathBuf::from)
        {
            fs::create_dir_all(&configured_directory_path)
                .expect("the configured benchmark worker home should be created");
            return Self {
                directory_path: configured_directory_path,
                _temporary_directory_owner: None,
            };
        }
        let temporary_directory_owner = tempfile::tempdir()
            .expect("the temporary prefill benchmark worker home should be created");
        Self {
            directory_path: temporary_directory_owner.path().to_path_buf(),
            _temporary_directory_owner: Some(temporary_directory_owner),
        }
    }

    fn directory_path(&self) -> &Path {
        &self.directory_path
    }
}

pub(super) fn configured_prefill_qualification_model_directory() -> PathBuf {
    crate::common::configured_model_artifact_directory_by_id(PREFILL_QUALIFICATION_MODEL_ID)
}

pub(super) fn build_prefill_qualification_prompt(
    model_directory: &Path,
    target_prompt_tokens: usize,
) -> String {
    build_exact_model_prompt_content(
        model_directory,
        PREFILL_QUALIFICATION_SOURCE_DOCUMENT,
        "Read the following public-domain book excerpt, then provide one concise sentence assessing its central conflict.",
        target_prompt_tokens,
    )
}

pub(super) fn prepare_prefill_qualification_worker(
    model_directory: &Path,
    fixed_prefill_chunck_tokens: Option<u32>,
    maximum_mlx_memory_gb: Option<u64>,
) -> PreparedPrefillQualificationWorker {
    let production_worker_executable_path =
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production worker executable path");
    let isolated_worker_home =
        IsolatedPrefillQualificationWorkerHome::from_environment_or_temporary();
    let configuration_directory = isolated_worker_home.directory_path().join(".astronomical");
    let optimizer_state_file_path = configuration_directory
        .join("optimizer")
        .join("prefill-chunck-size.json");
    let optimizer_state_loaded_before_run = optimizer_state_file_path.is_file();
    fs::create_dir_all(&configuration_directory)
        .expect("the isolated benchmark configuration directory should be created");
    let mut configuration_document = json!({
        "model_directories": [model_directory],
    });
    match fixed_prefill_chunck_tokens {
        Some(configured_fixed_prefill_chunck_tokens) => {
            configuration_document["prefill_chunck_size_optimizer_enabled"] = json!(false);
            configuration_document["fixed_prefill_chunck_tokens"] =
                json!(configured_fixed_prefill_chunck_tokens);
        }
        None => {
            configuration_document["prefill_chunck_size_optimizer_enabled"] = json!(true);
        }
    }
    if let Some(configured_maximum_mlx_memory_gb) = maximum_mlx_memory_gb {
        configuration_document["maximum_mlx_memory_gb"] = json!(configured_maximum_mlx_memory_gb);
    }
    if std::env::var_os("ASTRONOMICAL_PREFILL_QUALIFICATION_PERFORMANCE_ATTRIBUTION").is_some() {
        configuration_document["performance_attribution_enabled"] = json!(true);
    }
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the benchmark configuration should serialize"),
    )
    .expect("the benchmark configuration should be written");
    PreparedPrefillQualificationWorker {
        isolated_worker_home,
        worker_executable_path: PathBuf::from(production_worker_executable_path),
        optimizer_state_loaded_before_run,
    }
}

pub(super) async fn launch_prepared_prefill_qualification_worker(
    prepared_prefill_qualification_worker: &PreparedPrefillQualificationWorker,
    model_directory: &Path,
) -> WorkerHandle {
    let performance_log_directory = prepared_prefill_qualification_worker
        .isolated_worker_home_directory()
        .join("logs");
    fs::create_dir_all(&performance_log_directory)
        .expect("the prefill benchmark performance log directory should be created");
    let worker_runtime_config = ResolvedRuntimeConfigResolver::new(
        prepared_prefill_qualification_worker
            .isolated_worker_home_directory()
            .to_path_buf(),
        prepared_prefill_qualification_worker
            .worker_executable_path
            .clone(),
    )
    .load()
    .expect("the prefill benchmark worker configuration should resolve");
    WorkerHandle::launch_with_startup_configuration(
        prepared_prefill_qualification_worker
            .worker_executable_path
            .clone(),
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the prefill benchmark performance log should open"),
        crate::common::single_model_directories(PREFILL_QUALIFICATION_MODEL_ID, model_directory),
        u32::from(PREFILL_QUALIFICATION_MAXIMUM_OUTPUT_TOKENS),
        worker_runtime_config.worker_startup_configuration(),
    )
    .await
    .expect("the benchmark worker should launch")
}

pub(super) fn required_prefill_qualification_u32(
    environment_variable_name: &str,
    allowed_values: &[u32],
) -> u32 {
    let configured_value = std::env::var(environment_variable_name)
        .unwrap_or_else(|_| panic!("set {environment_variable_name} for the qualification cell"));
    let parsed_value = configured_value.parse::<u32>().unwrap_or_else(|_| {
        panic!("{environment_variable_name} must contain one integer, got {configured_value}")
    });
    assert!(
        allowed_values.contains(&parsed_value),
        "{environment_variable_name} must be one of {allowed_values:?}, got {parsed_value}"
    );
    parsed_value
}

pub(super) async fn wait_until_prefill_qualification_worker_is_idle(
    worker_handle: &WorkerHandle,
    worker_label: &str,
) {
    for readiness_attempt in 1..=EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if worker_health_snapshot.status == WorkerHealthStatus::Ready
            && worker_health_snapshot.ready_model_id.is_none()
        {
            eprintln!(
                "[prefill-optimizer:{worker_label}] idle worker ready after {readiness_attempt} attempts"
            );
            return;
        }
        let remaining_seconds =
            EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT.saturating_sub(readiness_attempt);
        eprintln!(
            "[prefill-optimizer:{worker_label}] startup attempt={readiness_attempt}/{EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT} ETA<={remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the benchmark worker did not become idle before its deadline");
}

pub(super) async fn warm_prefill_qualification_worker(
    worker_handle: &WorkerHandle,
    warm_generation_command: ChatGenerationCommand,
    warm_request_maximum_output_tokens: u16,
) {
    eprintln!("[prefill-optimizer:warmup] starting unmeasured warm request");
    let mut warm_stream_receiver = worker_handle
        .start_chat_generation(warm_generation_command)
        .await
        .expect("the prefill benchmark warm request should start");
    while let Some(warm_stream_event) = warm_stream_receiver.recv().await {
        match warm_stream_event {
            ChatGenerationStreamEvent::Completed {
                generated_token_count,
                ..
            } => {
                assert_eq!(
                    generated_token_count, warm_request_maximum_output_tokens,
                    "the warm request should emit its one requested token"
                );
                eprintln!("[prefill-optimizer:warmup] completed");
                return;
            }
            ChatGenerationStreamEvent::Failed { reason } => {
                panic!("the prefill benchmark warm request failed: {reason:?}");
            }
            ChatGenerationStreamEvent::Error(error_code) => {
                panic!("the prefill benchmark warm stream failed: {error_code:?}");
            }
            ChatGenerationStreamEvent::ReasoningFragment(_)
            | ChatGenerationStreamEvent::TextFragment(_)
            | ChatGenerationStreamEvent::ToolCall { .. }
            | ChatGenerationStreamEvent::PrefillProgress { .. } => {}
        }
    }
    panic!("the prefill benchmark warm stream closed before completion");
}

pub(super) fn prefill_memory_limit_validation_error(
    prefill_measurements: &PrefillMeasurementAccumulator,
    maximum_mlx_memory_gb: Option<u64>,
) -> Option<String> {
    let Some(maximum_mlx_memory_gb) = maximum_mlx_memory_gb else {
        return None;
    };
    let configured_maximum_mlx_memory_bytes = maximum_mlx_memory_gb.saturating_mul(1_000_000_000);
    if let Some(observed_stable_mlx_memory_bytes) = prefill_measurements
        .chuncks()
        .iter()
        .map(|prefill_chunck_measurement| prefill_chunck_measurement.mlx_active_memory_bytes)
        .find(|observed_active_memory_bytes| {
            *observed_active_memory_bytes > configured_maximum_mlx_memory_bytes
        })
    {
        return Some(format!(
            "prefill stable MLX memory exceeded the configured ceiling: active_bytes={observed_stable_mlx_memory_bytes}, configured_ceiling_bytes={configured_maximum_mlx_memory_bytes}"
        ));
    }
    let observed_peak_mlx_memory_bytes = prefill_measurements
        .chuncks()
        .iter()
        .map(|prefill_chunck_measurement| prefill_chunck_measurement.mlx_peak_memory_bytes)
        .max()
        .expect("the qualified prefill request should report MLX peak memory");
    let qualification_peak_limit_bytes = configured_maximum_mlx_memory_bytes.saturating_add(
        configured_maximum_mlx_memory_bytes
            / PREFILL_QUALIFICATION_TRANSIENT_MLX_PEAK_ALLOWANCE_DIVISOR,
    );
    (observed_peak_mlx_memory_bytes > qualification_peak_limit_bytes).then(|| {
        format!(
            "prefill MLX peak exceeded the qualification allowance: peak_bytes={observed_peak_mlx_memory_bytes}, configured_ceiling_bytes={configured_maximum_mlx_memory_bytes}, qualification_peak_limit_bytes={qualification_peak_limit_bytes}"
        )
    })
}

#[tokio::test]
async fn should_enforce_stable_and_transient_prefill_mlx_memory_limits() {
    tokio::time::timeout(Duration::from_secs(1), async {
        let mut prefill_measurements = PrefillMeasurementAccumulator::new();
        prefill_measurements.record(
            4_096,
            1,
            Some(1),
            Some(4_096),
            Some(10_000_000_000),
            Some(0),
            Some(10_061_330_796),
        );

        assert_eq!(
            prefill_memory_limit_validation_error(&prefill_measurements, Some(10)),
            None,
            "the measured 0.613 percent transient MLX peak should qualify"
        );
        prefill_measurements.record(
            8_192,
            2,
            Some(1),
            Some(4_096),
            Some(10_000_000_000),
            Some(0),
            Some(10_100_000_000),
        );
        assert_eq!(
            prefill_memory_limit_validation_error(&prefill_measurements, Some(10)),
            None,
            "a peak exactly at the one percent allowance should qualify"
        );
        prefill_measurements.record(
            12_288,
            3,
            Some(1),
            Some(4_096),
            Some(10_000_000_000),
            Some(0),
            Some(10_100_000_001),
        );
        assert!(
            prefill_memory_limit_validation_error(&prefill_measurements, Some(10)).is_some(),
            "the first byte beyond the one percent allowance must fail qualification"
        );
        let mut stable_memory_measurements = PrefillMeasurementAccumulator::new();
        stable_memory_measurements.record(
            4_096,
            1,
            Some(1),
            Some(4_096),
            Some(10_000_000_001),
            Some(0),
            Some(10_000_000_001),
        );
        assert!(
            prefill_memory_limit_validation_error(&stable_memory_measurements, Some(10))
                .is_some_and(|validation_error| validation_error.contains("stable MLX memory")),
            "stable active memory one byte above C must fail qualification"
        );
    })
    .await
    .expect("the prefill peak qualification contract should finish within one second");
}

pub(super) async fn observed_final_expert_memory_mode(
    worker_handle: &WorkerHandle,
    benchmark_label: &str,
) -> ExpertMemoryMode {
    for readiness_attempt in 1..=EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT {
        let worker_health_snapshot = worker_handle.worker_health_snapshot();
        if let Some(observed_expert_memory_mode) = worker_health_snapshot.expert_memory_mode {
            return observed_expert_memory_mode;
        }
        eprintln!(
            "[prefill-optimizer:{benchmark_label}] waiting for observed expert-memory mode attempt={readiness_attempt}/{EXPERT_MEMORY_MODE_READINESS_ATTEMPT_LIMIT}"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the benchmark worker did not report an observed expert-memory mode");
}

pub(super) const fn expert_memory_mode_label(expert_memory_mode: ExpertMemoryMode) -> &'static str {
    match expert_memory_mode {
        ExpertMemoryMode::Resident => "resident",
        ExpertMemoryMode::Paged => "paged",
    }
}
