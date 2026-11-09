//! Cold SSD-streaming prefill as production serves it, in process.
//!
//! Performance attribution synchronizes the graphics processor after every
//! prefill feed-forward stage, which changes how expert reads overlap and how
//! much memory a chunk holds at once. Measured 2026-11-09 on the 4-bit 35B
//! Atlas at 11 GiB with 4,096-token chunks: attributed prefill read about
//! 14 GB per chunk, the attribution-free REST worker about 22.6 GB. This
//! journey serves the same requests with attribution off, through the
//! in-process engine, so the storage reads behind a production prefill can be
//! examined directly. The model's weight files are kept out of the file
//! cache (`cold_file_cache`), as on a Mac whose RAM is the limit.
//!
//! Run it with `cargo measure-unattributed-cold-prefill --configuration PATH`.

use std::path::Path;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxMemoryLimits;
use astronomical_test_harness::test_run::TestRunContext;
use serial_test::serial;
use tokio::time::timeout;

use super::cold_file_cache::ColdFileCache;
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, RequestAttribution, create_attributed_engine, load_engine_with_progress,
    run_generation_with_request_attribution,
};

const WARMUP_REQUEST_ID: RequestId = RequestId::new(97_401);
const WARMUP_PROMPT_TOKEN_COUNT: usize = 500;
const WARMUP_OUTPUT_TOKEN_COUNT: u16 = 50;
const MEASURED_REQUEST_ID: RequestId = RequestId::new(97_402);
const MEASURED_PROMPT_TOKEN_COUNT: usize = 10_240;
/// Two tokens: enough to finish the prefill and take the decode handoff, so
/// the measured time is prefill.
const MEASURED_OUTPUT_TOKEN_COUNT: u16 = 2;
const PREFILL_JOURNEY_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "serves a 10,240-token Shakespeare prefill without attribution on an installed sparse model with a cold file cache"]
#[serial]
async fn should_measure_cold_prefill_without_attribution() {
    timeout(PREFILL_JOURNEY_TIMEOUT, measure_cold_prefill())
        .await
        .expect("the unattributed prefill journey must finish within 120 seconds");
}

async fn measure_cold_prefill() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_run_configuration = TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model");
    let paging_experiment = test_run_configuration.paging_experiment;
    let mlx_limit_bytes = usize::try_from(
        paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    let attribution_directory =
        tempfile::tempdir().expect("the journey should create an attribution directory");
    let mlx_memory_limits = MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
        .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_directory.path().join("unused.jsonl"),
        &mlx_memory_limits,
        paging_experiment.streaming_prompt_processing_chunk_tokens,
    );
    load_engine_with_progress(&mut engine, "unattributed_prefill_model_load").await;
    let cold_file_cache = ColdFileCache::start(&model_directory);
    for (request_id, prompt_token_count, output_token_count, phase_name) in [
        (
            WARMUP_REQUEST_ID,
            WARMUP_PROMPT_TOKEN_COUNT,
            WARMUP_OUTPUT_TOKEN_COUNT,
            "unattributed_prefill_warmup",
        ),
        (
            MEASURED_REQUEST_ID,
            MEASURED_PROMPT_TOKEN_COUNT,
            MEASURED_OUTPUT_TOKEN_COUNT,
            "unattributed_prefill_measured",
        ),
    ] {
        eprintln!("[unattributed-prefill] begin={phase_name}");
        let request_started_at = Instant::now();
        let generated_token_ids = run_generation_with_request_attribution(
            &mut engine,
            request_id,
            &shakespeare_prompt(
                &model_directory,
                request_id,
                prompt_token_count,
                output_token_count,
            ),
            phase_name,
            output_token_count,
            &end_of_sequence_token_ids,
            GenerationSampling::Seeded,
            RequestAttribution::Disabled,
        )
        .await;
        let request_seconds = request_started_at.elapsed().as_secs_f64();
        assert!(
            !generated_token_ids.is_empty(),
            "{phase_name} must generate"
        );
        eprintln!(
            "[unattributed-prefill] end={phase_name} prompt_tokens={prompt_token_count} request_seconds={request_seconds:.2} prompt_tokens_per_second={:.1}",
            prompt_token_count as f64 / request_seconds
        );
    }
    eprintln!(
        "[unattributed-prefill] file_cache_sweeps={}",
        cold_file_cache.finish()
    );
}

fn shakespeare_prompt(
    model_directory: &Path,
    request_id: RequestId,
    prompt_token_count: usize,
    output_token_count: u16,
) -> Vec<u32> {
    let model_id = model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name");
    crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        model_directory,
        model_id,
        request_id,
        prompt_token_count,
        output_token_count,
    )
}
