//! Storage-transition cells of the pinned `astro_ssd_prompt_cache` interaction matrix.
//!
//! Each ignored serial test names its storage transition: `LiveReuse`,
//! `WorkerRestart`, or `DeletedWhileLive`. Select exactly one full test name
//! per process through `cargo xtask test-run --package astronomical-acceptance
//! --test engine_astro_ssd_prompt_cache --features direct-mlx --release --configuration
//! PATH --filter <full_test_name> --exact --ignored`. There is no default cell.

use std::{fs, path::Path, time::Duration};

use ipc_protocol::RequestId;
use model_serving::{
    InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5TextEngine, Qwen3_5Tokenizer,
    TextInferenceRequest,
};
use serial_test::serial;
use tokio::time::timeout;

use super::engine_astro_ssd_prompt_cache::{
    generate_token_ids, load_astro_ssd_prompt_cache_acceptance_engine,
    wait_for_astro_ssd_prompt_cache_blocks,
};
use super::large_prefill_prompt::representative_long_generation_prompt_token_ids_capped_at;

const ASTRO_SSD_PROMPT_CACHE_INTERACTION_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(115);
const FIXED_PREFILL_CHUNK_TOKENS: u32 = 2_048;
/// Enough tokens that a state difference has room to show; one token proves little.
const COMPARED_OUTPUT_TOKEN_COUNT: u16 = 64;
const GREEDY_TEMPERATURE_THOUSANDTHS: u16 = 0;
const FULL_TOP_P_THOUSANDTHS: u16 = 1_000;

#[derive(Clone, Copy)]
enum StorageTransition {
    LiveReuse,
    WorkerRestart,
    DeletedWhileLive,
}

impl StorageTransition {
    fn label(self) -> &'static str {
        match self {
            Self::LiveReuse => "LiveReuse",
            Self::WorkerRestart => "WorkerRestart",
            Self::DeletedWhileLive => "DeletedWhileLive",
        }
    }
}

#[tokio::test]
#[ignore = "loads Ornith and proves the LiveReuse astro SSD prompt cache interaction matrix cell"]
#[serial]
async fn should_run_pinned_ornith_astro_ssd_prompt_cache_interaction_live_reuse() {
    run_selected_acceptance_cell(StorageTransition::LiveReuse).await;
}

#[tokio::test]
#[ignore = "loads Ornith and proves the WorkerRestart astro SSD prompt cache interaction matrix cell"]
#[serial]
async fn should_run_pinned_ornith_astro_ssd_prompt_cache_interaction_worker_restart() {
    run_selected_acceptance_cell(StorageTransition::WorkerRestart).await;
}

#[tokio::test]
#[ignore = "loads Ornith and proves the DeletedWhileLive astro SSD prompt cache interaction matrix cell"]
#[serial]
async fn should_run_pinned_ornith_astro_ssd_prompt_cache_interaction_deleted_while_live() {
    run_selected_acceptance_cell(StorageTransition::DeletedWhileLive).await;
}

async fn run_selected_acceptance_cell(storage_transition: StorageTransition) {
    eprintln!(
        "[astro-ssd-prompt-cache-interaction-matrix] status=start cell={} timeout_seconds={}",
        storage_transition.label(),
        ASTRO_SSD_PROMPT_CACHE_INTERACTION_ACCEPTANCE_TIMEOUT.as_secs()
    );
    timeout(
        ASTRO_SSD_PROMPT_CACHE_INTERACTION_ACCEPTANCE_TIMEOUT,
        run_acceptance_cell(storage_transition),
    )
    .await
    .expect("the selected astro-ssd-prompt-cache interaction matrix cell should finish within 115 seconds");
    eprintln!(
        "[astro-ssd-prompt-cache-interaction-matrix] status=success cell={}",
        storage_transition.label()
    );
}

async fn run_acceptance_cell(storage_transition: StorageTransition) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_resident_sparse_moe_model_directory();
    let astro_ssd_prompt_cache_directory = tempfile::tempdir()
        .expect("the acceptance should create an isolated astro SSD prompt cache directory");
    let (mut qwen3_5_engine, _, _, astro_ssd_prompt_cache_block_token_count) =
        load_astro_ssd_prompt_cache_acceptance_engine(
            &model_directory,
            astro_ssd_prompt_cache_directory.path(),
            FIXED_PREFILL_CHUNK_TOKENS,
        )
        .await;
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(
        &model_directory,
        astro_ssd_prompt_cache_block_token_count * 2 + 16,
    );

    let (cold_astro_ssd_prompt_cache_restored_token_count, cold_generated_token_ids) =
        run_greedy_request(
            &mut qwen3_5_engine,
            RequestId::new(30_001),
            &prompt_token_ids,
        )
        .await;
    assert_eq!(
        cold_astro_ssd_prompt_cache_restored_token_count, 0,
        "the request against an empty astro SSD prompt cache must miss"
    );
    wait_for_astro_ssd_prompt_cache_blocks(&qwen3_5_engine, 2).await;

    match storage_transition {
        StorageTransition::LiveReuse => {}
        StorageTransition::WorkerRestart => {
            drop(qwen3_5_engine);
            (qwen3_5_engine, _, _, _) = load_astro_ssd_prompt_cache_acceptance_engine(
                &model_directory,
                astro_ssd_prompt_cache_directory.path(),
                FIXED_PREFILL_CHUNK_TOKENS,
            )
            .await;
        }
        StorageTransition::DeletedWhileLive => {
            fs::remove_dir_all(astro_ssd_prompt_cache_directory.path())
                .expect("the acceptance should delete the astro SSD prompt cache while the engine remains live");
            let (
                replacement_cold_astro_ssd_prompt_cache_restored_token_count,
                replacement_cold_generated_token_ids,
            ) = run_greedy_request(
                &mut qwen3_5_engine,
                RequestId::new(30_002),
                &prompt_token_ids,
            )
            .await;
            assert_eq!(
                replacement_cold_astro_ssd_prompt_cache_restored_token_count, 0,
                "the first request after astro SSD prompt cache deletion must fall back to cold prefill"
            );
            assert_eq!(
                replacement_cold_generated_token_ids, cold_generated_token_ids,
                "astro SSD prompt cache deletion recovery must preserve deterministic output"
            );
            wait_for_astro_ssd_prompt_cache_blocks(&qwen3_5_engine, 2).await;
        }
    }

    let warm_request_id = match storage_transition {
        StorageTransition::DeletedWhileLive => RequestId::new(30_003),
        StorageTransition::LiveReuse | StorageTransition::WorkerRestart => RequestId::new(30_002),
    };
    let (warm_astro_ssd_prompt_cache_restored_token_count, warm_generated_token_ids) =
        run_greedy_request(&mut qwen3_5_engine, warm_request_id, &prompt_token_ids).await;
    assert!(
        warm_astro_ssd_prompt_cache_restored_token_count
            >= (astro_ssd_prompt_cache_block_token_count * 2) as u32,
        "the pre-existing astro SSD prompt cache restored {warm_astro_ssd_prompt_cache_restored_token_count} tokens but must restore both complete prompt blocks"
    );
    assert_eq!(
        warm_generated_token_ids, cold_generated_token_ids,
        "cold and restored deterministic output must match"
    );
}

fn romeo_and_juliet_prompt_token_ids(
    model_directory: &Path,
    target_prompt_token_count: usize,
) -> Vec<u32> {
    let prompt_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the selected astro SSD prompt cache interaction artifact should validate");
    let prompt_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&prompt_artifact)
        .expect("the selected astro SSD prompt cache interaction tokenizer should load");
    representative_long_generation_prompt_token_ids_capped_at(
        &prompt_tokenizer,
        prompt_artifact.model_id(),
        target_prompt_token_count,
    )
}

async fn run_greedy_request(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    request_id: RequestId,
    prompt_token_ids: &[u32],
) -> (u32, Vec<u32>) {
    let generation_start = qwen3_5_engine
        .start_generation(
            // `new` samples at temperature 1.0 with no seed, so two runs would compare two
            // random draws. Greedy makes the token a function of the logits alone, so a
            // changed token means the restored `astro_ssd_prompt_cache` changed them.
            TextInferenceRequest::new_sampling(
                request_id,
                prompt_token_ids.to_vec(),
                COMPARED_OUTPUT_TOKEN_COUNT,
                GREEDY_TEMPERATURE_THOUSANDTHS,
                FULL_TOP_P_THOUSANDTHS,
                None,
            )
            .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the acceptance engine should accept the request");
    let astro_ssd_prompt_cache_restored_token_count =
        generation_start.astro_ssd_prompt_cache_restored_token_count();
    let (generated_token_ids, _) = generate_token_ids(
        qwen3_5_engine,
        request_id,
        usize::from(COMPARED_OUTPUT_TOKEN_COUNT),
    )
    .await;
    // An end-of-sequence token may stop early, so the length is bounded rather than
    // fixed; the caller compares cold and warm answers token for token, length included.
    assert!(
        !generated_token_ids.is_empty()
            && generated_token_ids.len() <= usize::from(COMPARED_OUTPUT_TOKEN_COUNT),
        "a greedy request must emit between one and {COMPARED_OUTPUT_TOKEN_COUNT} tokens"
    );
    (
        astro_ssd_prompt_cache_restored_token_count,
        generated_token_ids,
    )
}
