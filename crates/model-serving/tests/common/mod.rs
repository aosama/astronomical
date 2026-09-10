#[allow(dead_code)]
mod e2e_test_model_names;
#[allow(dead_code, unused_imports)]
pub(crate) use e2e_test_model_names::{
    dense_mtp_model_id, e2e_test_model_ids, flux2_klein_model_id, k2_horizon_mova_model_id,
    laguna_xs_model_id, large_sparse_moe_model_id, required_e2e_test_model_ids,
    resident_sparse_moe_model_id, small_dense_model_id,
};

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
pub(crate) mod laguna;

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn test_worker_kernel_capabilities(
    runtime: &astronomical_runtime_integration::MlxRuntime,
) -> &'static astronomical_model_serving::WorkerKernelCapabilities {
    use astronomical_model_serving::{PerformanceAttribution, worker_process_kernel_capabilities};
    worker_process_kernel_capabilities(runtime, &mut PerformanceAttribution::disabled())
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn forced_unsupported_worker_kernel_capabilities()
-> astronomical_model_serving::WorkerKernelCapabilities {
    // Every family the Qwen3.5 loader consults is demoted with a distinct
    // unsupported reason so one serving request exercises each fallback
    // route; the K2-only fused expert decode stays unprobed and fail-closed.
    use astronomical_model_serving::{CustomKernelVerdict, CustomMetalKernelFamily};
    let forced_demotion_description = "forced demotion for the serving fallback journey";
    astronomical_model_serving::WorkerKernelCapabilities::with_forced_verdicts_for_tests([
        (
            CustomMetalKernelFamily::SortedExpertWeightedSum,
            CustomKernelVerdict::Unsupported(
                astronomical_model_serving::KernelUnsupportedReason::Compilation {
                    description: forced_demotion_description.to_owned(),
                },
            ),
        ),
        (
            CustomMetalKernelFamily::GatedDeltaSequence,
            CustomKernelVerdict::Unsupported(
                astronomical_model_serving::KernelUnsupportedReason::Execution {
                    description: forced_demotion_description.to_owned(),
                },
            ),
        ),
        (
            CustomMetalKernelFamily::GatedDeltaBoundaryCheckpoint,
            CustomKernelVerdict::Unsupported(
                astronomical_model_serving::KernelUnsupportedReason::OutputMismatch {
                    description: forced_demotion_description.to_owned(),
                },
            ),
        ),
        (
            CustomMetalKernelFamily::TargetVerificationQuantizedLinear,
            CustomKernelVerdict::Unsupported(
                astronomical_model_serving::KernelUnsupportedReason::OutputMismatch {
                    description: forced_demotion_description.to_owned(),
                },
            ),
        ),
        (
            CustomMetalKernelFamily::TargetVerificationFourRowQuantizedLinear,
            CustomKernelVerdict::Unsupported(
                astronomical_model_serving::KernelUnsupportedReason::OutputMismatch {
                    description: forced_demotion_description.to_owned(),
                },
            ),
        ),
    ])
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn standard_worker_chunking_configuration()
-> astronomical_ipc_protocol::WorkerChunkingConfiguration {
    astronomical_ipc_protocol::WorkerChunkingConfiguration {
        fixed_prompt_processing_chunk_size_tokens: 2_048,
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
        full_attention_key_value_growth_tokens: 256,
        speculative_prefill_draft_forward_tokens: 2_048,
        prefill_graph_submission_layer_interval: 0,
        experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
        experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
        prompt_cache_block_tokens: None,
        prompt_cache_common_prefix_stride_blocks: 4,
        experimental_decode_stage_attribution_enabled: false,
        experimental_quantized_kv_cache_enabled: false,
        experimental_fused_moe_decode_enabled: false,
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
        selection_chunk_token_count: 32,
        mandatory_trailing_token_count: 512,
        lookahead_token_count: 8,
        importance_pooling_kernel_token_count: 13,
    }
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn standard_qwen3_5_model_chunking_configuration()
-> astronomical_model_serving::Qwen3_5ModelChunkingConfiguration {
    astronomical_model_serving::Qwen3_5ModelChunkingConfiguration::new(256, 0, 1, 3, 2_048)
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
pub(crate) fn resolve_serving_acceptance_mlx_memory_ceiling_bytes(
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
pub(crate) async fn sample_serving_acceptance_mlx_memory_limits() -> MlxMemoryLimits {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model acceptance");
    let machine_mlx_memory_ceiling_bytes = sample_machine_mlx_memory_ceiling_bytes().await;
    let configured_mlx_memory_ceiling_bytes = astronomical_config
        .maximum_mlx_memory_bytes()
        .expect("the configured model-artifact MLX memory ceiling should be valid");
    let effective_mlx_memory_ceiling_bytes = resolve_serving_acceptance_mlx_memory_ceiling_bytes(
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

/// A temporary MLX memory policy installed on the shared process runtime and restored on drop.
///
/// MLX memory limits are process-global (one Metal device, one policy), and the serving
/// invariant is one policy per process — the daemon restarts the worker instead of
/// reconfiguring. Journeys that legitimately need a distinct policy (for example a cold
/// allocator cache) adopt the shared policy first and then transition through the production
/// live-transition seam (`update_memory_limits`), restoring it on both success and panic so
/// later journeys in the same test process keep composing.
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) struct MlxMemoryPolicyGuard {
    runtime: astronomical_runtime_integration::MlxRuntime,
    shared_memory_limits: MlxMemoryLimits,
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
impl MlxMemoryPolicyGuard {
    /// Requires the caller to already hold `direct_mlx_test_guard`, because the transition and
    /// restore both mutate the shared process policy.
    pub(crate) fn install_temporary_policy(
        shared_memory_limits: MlxMemoryLimits,
        temporary_memory_limits: MlxMemoryLimits,
    ) -> Self {
        let mut runtime =
            astronomical_runtime_integration::MlxRuntime::initialize(shared_memory_limits)
                .expect("the shared MLX memory policy should initialize for the journey");
        runtime
            .update_memory_limits(temporary_memory_limits)
            .expect("the journey should transition to its temporary MLX memory policy");
        Self {
            runtime,
            shared_memory_limits,
        }
    }
}

#[cfg(feature = "direct-mlx")]
impl Drop for MlxMemoryPolicyGuard {
    fn drop(&mut self) {
        // A journey that panicked mid-generation must still hand the shared process policy back
        // intact, or every later journey in the same binary fails on a policy it never chose.
        let _restore_result = self.runtime.update_memory_limits(self.shared_memory_limits);
        let _cache_release_result = self
            .runtime
            .synchronize_gpu_stream_and_reclaim_allocator_cache_above_threshold(0);
    }
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
/// Uses only the machine ceiling so residency acceptance is not changed by
/// a developer's ordinary lower application cap.
pub(crate) async fn sample_machine_serving_acceptance_mlx_memory_limits() -> MlxMemoryLimits {
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
pub(crate) fn configured_large_sparse_moe_model_directory() -> PathBuf {
    configured_installed_model_directory_by_id(large_sparse_moe_model_id())
}

#[cfg(feature = "direct-mlx")]
/// Resolves the executable models exactly as the supervisor daemon serves them: the automatic
/// Library destination takes precedence over the authored config roots. Using this instead of
/// raw `discover_models(config.model_directories())` keeps acceptance journeys consistent with
/// production even when an artifact exists only under the instance state models directory.
#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn effective_discovered_models(
    astronomical_config: &AstronomicalConfig,
) -> Vec<astronomical_config::DiscoveredModel> {
    let instance_paths =
        astronomical_config::AstronomicalInstancePaths::default_location_instance_paths(
            astronomical_config::AstronomicalRuntimeInstance::Development,
        )
        .expect("the standard Astronomical instance paths should resolve");
    astronomical_config::discover_effective_models(
        &instance_paths.models_directory(),
        astronomical_config.model_directories(),
    )
    .expect("effective model discovery should complete")
    .discovered_models
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_discovered_model_by_id(
    astronomical_config: &AstronomicalConfig,
    model_id: &str,
) -> astronomical_config::DiscoveredModel {
    effective_discovered_models(astronomical_config)
        .into_iter()
        .find(|discovered_model| discovered_model.model_id == model_id)
        .unwrap_or_else(|| {
            panic!(
                "effective model discovery should find model ID {model_id} under the standard Astronomical configuration"
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
pub(crate) fn configured_installed_model_directory_by_id(model_id: &str) -> PathBuf {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model acceptance");
    configured_discovered_model_by_id(&astronomical_config, model_id).model_directory
}

#[cfg(feature = "direct-mlx")]
#[allow(dead_code)]
pub(crate) fn configured_model_directory_by_id(model_id: &str) -> Option<PathBuf> {
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for model acceptance");
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
        .expect("~/.astronomical-dev/config.json should load for model-artifact acceptance")
        .prompt_cache()
        .expect("~/.astronomical-dev/config.json should define prompt_cache.max_size_gb")
        .global_prompt_cache_maximum_size_bytes()
}
