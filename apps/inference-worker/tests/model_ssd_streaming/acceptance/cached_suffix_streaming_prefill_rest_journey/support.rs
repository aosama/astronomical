//! Status evidence and isolated configuration for the residency interaction journey.

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use serde_json::{Value, json};

use super::{
    LOG_MARKER, MAXIMUM_OUTPUT_TOKEN_COUNT, PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
    PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL, PREFILL_CHUNK_TOKEN_COUNT,
};

/// Builds realistic tool-control pressure without coupling qualification to one client.
pub(super) fn production_shaped_tools(tool_count: usize) -> Vec<Value> {
    (0..tool_count)
        .map(|tool_number| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("astronomical_qualification_tool_{tool_number}"),
                    "description": "A local qualification tool that must not be called.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false,
                    },
                },
            })
        })
        .collect()
}

pub(super) fn artifact_directory_regular_file_bytes(model_directory: &Path) -> u64 {
    let mut artifact_payload_bytes = 0_u64;
    let mut pending_directories = vec![model_directory.to_path_buf()];
    while let Some(current_directory) = pending_directories.pop() {
        let directory_entries = fs::read_dir(&current_directory).unwrap_or_else(|read_error| {
            panic!("the discovered model directory should be readable: {read_error}")
        });
        for directory_entry in directory_entries {
            let directory_entry = directory_entry.unwrap_or_else(|read_error| {
                panic!("a model directory entry should be readable: {read_error}")
            });
            let entry_path = directory_entry.path();
            let entry_metadata = directory_entry.metadata().unwrap_or_else(|read_error| {
                panic!("model file metadata should be readable: {read_error}")
            });
            if entry_metadata.is_dir() {
                pending_directories.push(entry_path);
            } else if entry_metadata.is_file() {
                artifact_payload_bytes =
                    artifact_payload_bytes.saturating_add(entry_metadata.len());
            }
        }
    }
    artifact_payload_bytes
}

pub(super) fn write_interaction_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    allocated_mlx_memory_bytes: u64,
) {
    let interaction_instance_paths = interaction_instance_paths(isolated_worker_home);
    let configuration_directory = interaction_instance_paths.state_directory();
    fs::create_dir(&configuration_directory)
        .expect("the interaction configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": allocated_mlx_memory_bytes / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": 50,
        "performance_attribution_enabled": true,
        "logging": {"level": "info", "retained_files": 2},
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": PREFILL_CHUNK_TOKEN_COUNT,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": PREFILL_CHUNK_TOKEN_COUNT,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the interaction configuration should serialize"),
    )
    .expect("the interaction configuration should be written");
}

/// Copies isolated reports to a durable directory before the temp home is dropped.
///
/// Disposable Cargo targets are deleted after the journey, so evidence cannot live
/// there. The workspace `target/` directory survives that cleanup and is gitignored.
/// `ASTRONOMICAL_QUALIFICATION_EVIDENCE_DIRECTORY` overrides the destination.
pub(super) fn persist_qualification_evidence(isolated_worker_home: &Path) -> PathBuf {
    let logging_directory = interaction_instance_paths(isolated_worker_home).logging_directory();
    let evidence_directory = qualification_evidence_directory();
    fs::create_dir_all(&evidence_directory)
        .expect("the qualification evidence directory should be created");
    for report_file_name in ["performance.jsonl", "performance-attribution.jsonl"] {
        let source_report_path = logging_directory.join(report_file_name);
        if source_report_path.exists() {
            fs::copy(
                &source_report_path,
                evidence_directory.join(report_file_name),
            )
            .unwrap_or_else(|copy_error| {
                panic!("qualification evidence should copy {report_file_name}: {copy_error}")
            });
        }
    }
    copy_worker_log_files(&logging_directory, &evidence_directory);
    print_worker_diagnostic_logs(&logging_directory);
    eprintln!(
        "{LOG_MARKER} request=journey status=evidence path={}",
        evidence_directory.display()
    );
    evidence_directory
}

pub(super) fn qualification_evidence_root() -> PathBuf {
    if let Ok(configured_evidence_directory) =
        env::var("ASTRONOMICAL_QUALIFICATION_EVIDENCE_DIRECTORY")
    {
        return PathBuf::from(configured_evidence_directory);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/qualification-evidence/cached-suffix-streaming-prefill")
}

fn qualification_evidence_directory() -> PathBuf {
    let unix_epoch_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the qualification clock should be after the Unix epoch")
        .as_millis();
    qualification_evidence_root().join(unix_epoch_millis.to_string())
}

pub(super) fn write_json_document(document_path: &Path, document: &Value) {
    if let Some(parent_directory) = document_path.parent() {
        fs::create_dir_all(parent_directory)
            .expect("the qualification summary directory should be created");
    }
    fs::write(
        document_path,
        serde_json::to_vec_pretty(document).expect("the qualification summary should serialize"),
    )
    .expect("the qualification summary should be written");
}

fn copy_worker_log_files(logging_directory: &Path, evidence_directory: &Path) {
    let Ok(directory_entries) = fs::read_dir(logging_directory) else {
        return;
    };
    for directory_entry in directory_entries.flatten() {
        let file_name = directory_entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with("worker") && file_name.ends_with(".log") {
            let _ = fs::copy(
                directory_entry.path(),
                evidence_directory.join(file_name.as_ref()),
            );
        }
    }
}

/// Prints worker lines that explain a stall: eligibility, budget, layer walk, OOM.
pub(super) fn print_worker_diagnostic_logs(logging_directory: &Path) {
    let Ok(directory_entries) = fs::read_dir(logging_directory) else {
        eprintln!("{LOG_MARKER} request=journey status=worker_logs reason=unreadable");
        return;
    };
    let mut worker_log_paths = directory_entries
        .flatten()
        .map(|directory_entry| directory_entry.path())
        .filter(|log_path| {
            log_path
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .is_some_and(|file_name| {
                    file_name.starts_with("worker") && file_name.ends_with(".log")
                })
        })
        .collect::<Vec<_>>();
    worker_log_paths.sort();
    if worker_log_paths.is_empty() {
        eprintln!("{LOG_MARKER} request=journey status=worker_logs reason=missing");
        return;
    }
    for worker_log_path in worker_log_paths {
        let Ok(log_contents) = fs::read_to_string(&worker_log_path) else {
            continue;
        };
        eprintln!(
            "{LOG_MARKER} request=journey status=worker_log path={}",
            worker_log_path.display()
        );
        for log_line in log_contents.lines() {
            if log_line.contains("leftover expert budget")
                || log_line.contains("published phase-aware")
                || log_line.contains("ActiveMemory")
                || log_line.contains("active-memory ceiling")
                || log_line.contains("capacity")
                || log_line.contains("reclaim")
                || log_line.contains("ERROR")
                || log_line.contains("error")
                || log_line.contains("FATAL")
                || log_line.contains("fatal")
                || log_line.contains("panic")
                || log_line.contains("absent from the streamed page")
                || log_line.contains("InvalidPaging")
                || log_line.contains("ExpertPaging")
            {
                eprintln!("{LOG_MARKER} request=journey status=worker_trace line={log_line}");
            }
        }
    }
}

/// Derives every mutable journey path through the same Development instance boundary.
pub(super) fn interaction_instance_paths(isolated_worker_home: &Path) -> AstronomicalInstancePaths {
    AstronomicalInstancePaths::for_home_directory(
        isolated_worker_home,
        AstronomicalRuntimeInstance::Development,
    )
}

#[derive(Default)]
pub(super) struct InteractionLiveEvidence {
    pub(super) observed_active: bool,
    pub(super) observed_prompt_processing: bool,
    pub(super) observed_generation_preparation: bool,
    pub(super) observed_generation_preparation_with_consistent_residency: bool,
    pub(super) observed_generation: bool,
    pub(super) maximum_active_memory_bytes: u64,
    pub(super) maximum_peak_memory_bytes: u64,
    pub(super) maximum_expert_payload_bytes: u64,
    pub(super) longest_unmoving_prefill_seconds: f64,
    pub(super) final_status: Value,
}

impl InteractionLiveEvidence {
    pub(super) fn observe(&mut self, status_document: &Value) {
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        self.observed_active |= activity != "idle";
        self.observed_prompt_processing |= activity == "prompt_processing";
        self.observed_generation_preparation |= activity == "generation_preparation";
        if activity == "generation_preparation" {
            let expert_residency = &status_document["expert_residency"];
            let resident_expert_count = expert_residency["resident_expert_count"]
                .as_u64()
                .unwrap_or(0);
            let resident_expert_payload_bytes = expert_residency["resident_expert_payload_bytes"]
                .as_u64()
                .unwrap_or(0);
            self.observed_generation_preparation_with_consistent_residency |=
                (resident_expert_count == 0) == (resident_expert_payload_bytes == 0);
        }
        self.observed_generation |= activity == "generating";
        let memory = MemorySample::from_status(status_document);
        self.maximum_active_memory_bytes = self
            .maximum_active_memory_bytes
            .max(memory.active_memory_bytes);
        self.maximum_peak_memory_bytes =
            self.maximum_peak_memory_bytes.max(memory.peak_memory_bytes);
        self.maximum_expert_payload_bytes = self
            .maximum_expert_payload_bytes
            .max(memory.expert_payload_bytes);
    }
}

#[derive(Default)]
pub(super) struct ProgressSample {
    pub(super) processed_tokens: u64,
    pub(super) elapsed_millis: u64,
    pub(super) request_elapsed_millis: u64,
    // Preserve each phase's latest rate while the other phase is active so every
    // live sample reports both user-visible throughput measurements explicitly.
    pub(super) prefill_tokens_per_second: f64,
    pub(super) generation_tokens_per_second: f64,
}

#[derive(Default)]
pub(super) struct MemorySample {
    pub(super) active_memory_bytes: u64,
    pub(super) allocator_cache_memory_bytes: u64,
    pub(super) peak_memory_bytes: u64,
    pub(super) expert_payload_bytes: u64,
    pub(super) model_core_payload_bytes: u64,
    pub(super) context_state_payload_bytes: u64,
    pub(super) runtime_work_payload_bytes: u64,
}

impl MemorySample {
    pub(super) fn from_status(status_document: &Value) -> Self {
        let memory = &status_document["mlx_memory_snapshot"];
        Self {
            active_memory_bytes: memory["active_memory_bytes"].as_u64().unwrap_or(0),
            allocator_cache_memory_bytes: memory["allocator_cache_memory_bytes"]
                .as_u64()
                .unwrap_or(0),
            peak_memory_bytes: memory["peak_memory_bytes"].as_u64().unwrap_or(0),
            expert_payload_bytes: memory["expert_payload_bytes"].as_u64().unwrap_or(0),
            model_core_payload_bytes: memory["model_core_payload_bytes"].as_u64().unwrap_or(0),
            context_state_payload_bytes: memory["context_state_payload_bytes"]
                .as_u64()
                .unwrap_or(0),
            runtime_work_payload_bytes: memory["runtime_work_payload_bytes"].as_u64().unwrap_or(0),
        }
    }
}
