//! The isolated worker configuration the sparse-artifact serving journeys write
//! before launching the production worker.

use std::{fs, path::Path};

use serde_json::json;

use crate::support::{
    artifact_bytes::acceptance_model_scan_root, marketed_gigabytes::marketed_gigabytes_at_least,
};

use super::sparse_artifact_residency::MAXIMUM_OUTPUT_TOKEN_COUNT;

const FIXED_PREFILL_CHUNK_SIZE_TOKENS: u32 = 2_048;

pub(super) fn write_acceptance_config(
    worker_home: &Path,
    model_directory: &Path,
    model_id: &str,
    mlx_memory_ceiling_bytes: u64,
    astro_ssd_prompt_cache_enabled: bool,
) {
    let configuration_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the isolated acceptance configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [acceptance_model_scan_root(model_directory)],
            "maximum_mlx_memory_gb": marketed_gigabytes_at_least(mlx_memory_ceiling_bytes),
        },
        "ssd_prompt_cache": {
            "enabled": astro_ssd_prompt_cache_enabled,
        },
        "diagnostics": {"performance_attribution_enabled": true},
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": FIXED_PREFILL_CHUNK_SIZE_TOKENS
        },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
                },
            },
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the acceptance configuration should serialize"),
    )
    .expect("the acceptance configuration should be written");
}
