#[cfg(feature = "direct-mlx")]
use std::path::PathBuf;
#[cfg(feature = "direct-mlx")]
use std::time::Duration;

#[cfg(feature = "direct-mlx")]
use astronomical_config::AstronomicalConfig;
#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::{
    MlxMemoryLimits, maximum_recommended_gpu_working_set_size_bytes,
};
#[cfg(feature = "direct-mlx")]
use tokio::sync::{Mutex, MutexGuard};
#[cfg(feature = "direct-mlx")]
use tokio::{process::Command, time::timeout};

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) mod generation_progress;
#[allow(dead_code)]
pub(crate) mod mtp_depth_release_gate;
#[allow(dead_code)]
pub(crate) mod qwen3_5;
pub(crate) mod qwen3_5_moe;

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn standard_worker_chunking_configuration()
-> astronomical_ipc_protocol::WorkerChunkingConfiguration {
    astronomical_ipc_protocol::WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
        full_attention_key_value_growth_tokens: 256,
        speculative_prefill_draft_forward_tokens: 2_048,
        prefill_graph_submission_layer_interval: 1,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
    }
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn disabled_worker_speculative_prefill_configuration()
-> astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration {
    astronomical_ipc_protocol::WorkerSpeculativePrefillConfiguration {
        enabled: false,
        target_model_id: None,
        draft_model_id: None,
        draft_model_directory: None,
        minimum_prompt_tokens: 8_192,
        keep_percentage: 20,
        selection_chunck_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    }
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn standard_qwen3_5_model_chunking_configuration()
-> astronomical_model_serving::Qwen3_5ModelChunkingConfiguration {
    astronomical_model_serving::Qwen3_5ModelChunkingConfiguration::new(256, 0, 3, 2_048)
        .expect("the standard test model chunking configuration should be valid")
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn standard_request_decoder_state(
    qwen3_5_config: &astronomical_model_serving::Qwen3_5Config,
) -> astronomical_model_serving::RequestDecoderStateStack {
    astronomical_model_serving::RequestDecoderStateStack::empty_from_config_with_full_attention_kv_state_growth_tokens(
        qwen3_5_config,
        256,
    )
    .expect("the standard test decoder-state growth should be valid")
}

#[allow(dead_code)]
pub(crate) fn resolve_model_artifact_qualification_mlx_memory_ceiling_bytes(
    configured_mlx_memory_ceiling_bytes: Option<u64>,
    machine_mlx_memory_ceiling_bytes: usize,
) -> usize {
    configured_mlx_memory_ceiling_bytes.map_or(
        machine_mlx_memory_ceiling_bytes,
        |configured_mlx_memory_ceiling_bytes| {
            usize::try_from(configured_mlx_memory_ceiling_bytes)
                .unwrap_or(usize::MAX)
                .min(machine_mlx_memory_ceiling_bytes)
        },
    )
}

#[allow(dead_code)]
pub(crate) const SYNTHETIC_RED_PNG_BYTES: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 240, 31, 0,
    5, 0, 1, 255, 137, 153, 61, 29, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
static DIRECT_MLX_TEST_LOCK: Mutex<()> = Mutex::const_new(());

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
const BYTES_PER_MEBIBYTE: usize = 1024 * 1024;
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
const IOGPU_WIRED_LIMIT_SYSCTL_KEY: &str = "iogpu.wired_limit_mb";
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
const SYSCTL_EXECUTABLE_PATH: &str = "/usr/sbin/sysctl";
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
const MODEL_ARTIFACT_MLX_MEMORY_LIMIT_SAMPLE_TIMEOUT: Duration = Duration::from_secs(2);

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) const DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) const DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024;

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) async fn direct_mlx_test_guard() -> MutexGuard<'static, ()> {
    DIRECT_MLX_TEST_LOCK.lock().await
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) async fn sample_model_artifact_qualification_mlx_memory_limits() -> MlxMemoryLimits {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model qualification");
    let machine_mlx_memory_ceiling_bytes = sample_machine_mlx_memory_ceiling_bytes().await;
    let configured_mlx_memory_ceiling_bytes = astronomical_config
        .maximum_mlx_memory_bytes()
        .expect("the configured model-artifact MLX memory ceiling should be valid");
    let effective_mlx_memory_ceiling_bytes =
        resolve_model_artifact_qualification_mlx_memory_ceiling_bytes(
            configured_mlx_memory_ceiling_bytes,
            machine_mlx_memory_ceiling_bytes,
        );
    eprintln!(
        "[model-artifact-memory] machine_mlx_memory_ceiling_bytes={} configured_mlx_memory_ceiling_bytes={:?} effective_mlx_memory_ceiling_bytes={} active_memory_limit_bytes={} allocator_cache_memory_limit_bytes={}",
        machine_mlx_memory_ceiling_bytes,
        configured_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
    );
    MlxMemoryLimits::new(
        effective_mlx_memory_ceiling_bytes,
        effective_mlx_memory_ceiling_bytes,
    )
    .expect("the machine-derived model-artifact MLX memory limits should be valid")
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
/// Uses only the machine ceiling so residency qualifications are not changed by
/// a developer's ordinary lower application cap.
pub(crate) async fn sample_machine_model_artifact_qualification_mlx_memory_limits()
-> MlxMemoryLimits {
    let machine_mlx_memory_ceiling_bytes = sample_machine_mlx_memory_ceiling_bytes().await;
    eprintln!(
        "[model-artifact-machine-memory] machine_mlx_memory_ceiling_bytes={} active_memory_limit_bytes={} allocator_cache_memory_limit_bytes={}",
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    );
    MlxMemoryLimits::new(
        machine_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes,
    )
    .expect("the machine model-artifact MLX memory limits should be valid")
}

#[cfg(feature = "direct-mlx")]
async fn sample_machine_mlx_memory_ceiling_bytes() -> usize {
    let mut sysctl_command = Command::new(SYSCTL_EXECUTABLE_PATH);
    sysctl_command
        .arg("-n")
        .arg(IOGPU_WIRED_LIMIT_SYSCTL_KEY)
        .kill_on_drop(true);
    let sysctl_output = timeout(
        MODEL_ARTIFACT_MLX_MEMORY_LIMIT_SAMPLE_TIMEOUT,
        sysctl_command.output(),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "sampling {IOGPU_WIRED_LIMIT_SYSCTL_KEY} should finish within {} seconds",
            MODEL_ARTIFACT_MLX_MEMORY_LIMIT_SAMPLE_TIMEOUT.as_secs()
        )
    })
    .unwrap_or_else(|sample_error| {
        panic!("should sample {IOGPU_WIRED_LIMIT_SYSCTL_KEY}: {sample_error}")
    });
    assert!(
        sysctl_output.status.success(),
        "sysctl should read {IOGPU_WIRED_LIMIT_SYSCTL_KEY} successfully"
    );
    let wired_limit_mebibytes_text = String::from_utf8_lossy(&sysctl_output.stdout);
    let wired_limit_mebibytes = wired_limit_mebibytes_text
        .trim()
        .parse::<usize>()
        .unwrap_or_else(|parse_error| {
            panic!("{IOGPU_WIRED_LIMIT_SYSCTL_KEY} should be an unsigned integer: {parse_error}")
        });
    if wired_limit_mebibytes == 0 {
        maximum_recommended_gpu_working_set_size_bytes()
            .expect("MLX should expose the default GPU wired-memory working set")
    } else {
        wired_limit_mebibytes
            .checked_mul(BYTES_PER_MEBIBYTE)
            .expect("the GPU wired-memory limit should fit in usize bytes")
    }
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_ornith_model_artifact_directory() -> PathBuf {
    configured_model_artifact_directory_by_id(
        astronomical_model_serving::ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
    )
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_discovered_model_by_id(
    astronomical_config: &AstronomicalConfig,
    model_id: &str,
) -> astronomical_config::DiscoveredModel {
    astronomical_config::discover_models(astronomical_config.model_directories())
        .unwrap_or_else(|discovery_error| {
            panic!(
                "model_directories discovery should complete for model ID {model_id}: {discovery_error}"
            )
        })
        .into_iter()
        .flat_map(|model_directory_scan| model_directory_scan.discovered_models)
        .find(|discovered_model| discovered_model.model_id == model_id)
        .unwrap_or_else(|| {
            panic!(
                "the standard Astronomical configuration model_directories should discover model ID {model_id}"
            )
        })
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn discovered_chat_capabilities(
    discovered_model: &astronomical_config::DiscoveredModel,
) -> &astronomical_config::ChatModelCapabilities {
    let astronomical_config::ModelCapabilities::Chat(chat_capabilities) =
        &discovered_model.capabilities
    else {
        panic!(
            "model ID {} should identify a discovered chat model",
            discovered_model.model_id
        );
    };
    chat_capabilities
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_model_artifact_directory_by_id(model_id: &str) -> PathBuf {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model qualification");
    configured_discovered_model_by_id(&astronomical_config, model_id).model_directory
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_model_directory_by_id(model_id: &str) -> Option<PathBuf> {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model qualification");
    astronomical_config
        .find_configured_model_directory_by_id(model_id)
        .unwrap_or_else(|discovery_error| {
            panic!(
                "model_directories discovery should complete for model ID {model_id}: {discovery_error}"
            )
        })
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_model_artifact_prompt_cache_maximum_size_bytes() -> u64 {
    AstronomicalConfig::load_from_development_location()
        .expect("~/.astronomical-dev/config.json should load for model-artifact qualification")
        .prompt_cache()
        .expect("~/.astronomical-dev/config.json should define prompt_cache.max_size_gb")
        .global_prompt_cache_maximum_size_bytes()
}
