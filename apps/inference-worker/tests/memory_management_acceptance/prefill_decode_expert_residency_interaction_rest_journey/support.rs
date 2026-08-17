//! Status evidence and isolated configuration for the residency interaction journey.

use std::{fs, path::Path};

use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};
use serde_json::{Value, json};

use super::{
    MAXIMUM_MLX_MEMORY_BYTES, MAXIMUM_OUTPUT_TOKEN_COUNT, PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL,
    PREFILL_CHUNCK_TOKEN_COUNT,
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

pub(super) fn write_interaction_config(isolated_worker_home: &Path, model_directory: &Path) {
    let interaction_instance_paths = interaction_instance_paths(isolated_worker_home);
    let configuration_directory = interaction_instance_paths.state_directory();
    fs::create_dir(&configuration_directory)
        .expect("the interaction configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": MAXIMUM_MLX_MEMORY_BYTES / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": 10,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "logging": {"level": "debug", "retained_files": 2},
        "chunking": {

            "fixed_prompt_processing_chunk_size_tokens": PREFILL_CHUNCK_TOKEN_COUNT,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": PAGING_GRAPH_SUBMISSION_LAYER_INTERVAL,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the interaction configuration should serialize"),
    )
    .expect("the interaction configuration should be written");
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
            let complete_layer_count = expert_residency["complete_layer_count"]
                .as_u64()
                .unwrap_or(0);
            let complete_layer_payload_bytes = expert_residency["complete_layer_payload_bytes"]
                .as_u64()
                .unwrap_or(0);
            let partial_layer_count = expert_residency["partial_layer_count"]
                .as_u64()
                .unwrap_or(0);
            let partial_layer_payload_bytes = expert_residency["partial_layer_payload_bytes"]
                .as_u64()
                .unwrap_or(0);
            // A layer count and its payload must describe the same ownership
            // snapshot; a nonzero count with zero bytes is not truthful status.
            self.observed_generation_preparation_with_consistent_residency |= complete_layer_count
                > 0
                && complete_layer_payload_bytes > 0
                && (partial_layer_count == 0) == (partial_layer_payload_bytes == 0);
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
