use std::{fs, time::Duration};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{InferenceEngine, Qwen3_5Engine, Qwen3_5InferenceRequest};
use tokio::time::timeout;

use super::engine_prompt_cache::{
    generate_token_ids, load_persistent_prompt_cache_qualification_engine,
    persistent_prompt_cache_eligible_prompt_token_ids, wait_for_persistent_prompt_cache_blocks,
};

const CACHE_INTERACTION_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const FIXED_PREFILL_CHUNCK_TOKENS: u32 = 4_096;

#[derive(Clone, Copy)]
enum StorageTransition {
    LiveReuse,
    WorkerRestart,
    DeletedWhileLive,
}

#[derive(Clone, Copy)]
struct CacheInteractionQualificationCell {
    environment_label: &'static str,
    storage_transition: StorageTransition,
}

const CACHE_INTERACTION_QUALIFICATION_CELLS: [CacheInteractionQualificationCell; 3] = [
    CacheInteractionQualificationCell {
        environment_label: "fixed-live-reuse",
        storage_transition: StorageTransition::LiveReuse,
    },
    CacheInteractionQualificationCell {
        environment_label: "fixed-worker-restart",
        storage_transition: StorageTransition::WorkerRestart,
    },
    CacheInteractionQualificationCell {
        environment_label: "fixed-deleted-while-live",
        storage_transition: StorageTransition::DeletedWhileLive,
    },
];

#[tokio::test]
#[ignore = "loads Ornith and qualifies one selected cache interaction matrix cell"]
async fn should_qualify_selected_pinned_ornith_cache_interaction_matrix_cell() {
    let selected_cell = selected_qualification_cell();
    eprintln!(
        "[prompt-cache-interaction-matrix] status=start cell={} timeout_seconds={}",
        selected_cell.environment_label,
        CACHE_INTERACTION_QUALIFICATION_TIMEOUT.as_secs()
    );
    timeout(
        CACHE_INTERACTION_QUALIFICATION_TIMEOUT,
        run_qualification_cell(selected_cell),
    )
    .await
    .expect("the selected prompt-cache interaction matrix cell should finish within 115 seconds");
    eprintln!(
        "[prompt-cache-interaction-matrix] status=success cell={}",
        selected_cell.environment_label
    );
}

fn selected_qualification_cell() -> CacheInteractionQualificationCell {
    let selected_environment_label =
        std::env::var("ASTRONOMICAL_PROMPT_CACHE_INTERACTION_QUALIFICATION_CELL")
            .expect("set ASTRONOMICAL_PROMPT_CACHE_INTERACTION_QUALIFICATION_CELL");
    CACHE_INTERACTION_QUALIFICATION_CELLS
        .iter()
        .copied()
        .find(|qualification_cell| {
            qualification_cell.environment_label == selected_environment_label
        })
        .unwrap_or_else(|| {
            let allowed_environment_labels = CACHE_INTERACTION_QUALIFICATION_CELLS
                .iter()
                .map(|qualification_cell| qualification_cell.environment_label)
                .collect::<Vec<_>>();
            panic!(
                "ASTRONOMICAL_PROMPT_CACHE_INTERACTION_QUALIFICATION_CELL must be one of {allowed_environment_labels:?}, got {selected_environment_label}"
            )
        })
}

async fn run_qualification_cell(qualification_cell: CacheInteractionQualificationCell) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the qualification should create an isolated cache directory");
    let (mut qwen3_5_engine, _, _, prompt_cache_block_token_count) =
        load_persistent_prompt_cache_qualification_engine(
            &model_directory,
            persistent_prompt_cache_directory.path(),
            FIXED_PREFILL_CHUNCK_TOKENS,
        )
        .await;
    let prompt_token_ids =
        persistent_prompt_cache_eligible_prompt_token_ids(prompt_cache_block_token_count * 2 + 16);

    let (cold_cached_token_count, cold_generated_token_ids) = run_one_token_request(
        &mut qwen3_5_engine,
        RequestId::new(30_001),
        &prompt_token_ids,
    )
    .await;
    assert_eq!(
        cold_cached_token_count, 0,
        "the empty-cache request must miss"
    );
    wait_for_persistent_prompt_cache_blocks(&qwen3_5_engine, 2).await;

    match qualification_cell.storage_transition {
        StorageTransition::LiveReuse => {}
        StorageTransition::WorkerRestart => {
            drop(qwen3_5_engine);
            (qwen3_5_engine, _, _, _) = load_persistent_prompt_cache_qualification_engine(
                &model_directory,
                persistent_prompt_cache_directory.path(),
                FIXED_PREFILL_CHUNCK_TOKENS,
            )
            .await;
        }
        StorageTransition::DeletedWhileLive => {
            fs::remove_dir_all(persistent_prompt_cache_directory.path())
                .expect("the qualification should delete the cache while the engine remains live");
            let (replacement_cold_cached_token_count, replacement_cold_generated_token_ids) =
                run_one_token_request(
                    &mut qwen3_5_engine,
                    RequestId::new(30_002),
                    &prompt_token_ids,
                )
                .await;
            assert_eq!(
                replacement_cold_cached_token_count, 0,
                "the first request after cache deletion must fall back to cold prefill"
            );
            assert_eq!(
                replacement_cold_generated_token_ids, cold_generated_token_ids,
                "cache deletion recovery must preserve deterministic output"
            );
            wait_for_persistent_prompt_cache_blocks(&qwen3_5_engine, 2).await;
        }
    }

    let warm_request_id = match qualification_cell.storage_transition {
        StorageTransition::DeletedWhileLive => RequestId::new(30_003),
        StorageTransition::LiveReuse | StorageTransition::WorkerRestart => RequestId::new(30_002),
    };
    let (warm_cached_token_count, warm_generated_token_ids) =
        run_one_token_request(&mut qwen3_5_engine, warm_request_id, &prompt_token_ids).await;
    assert!(
        warm_cached_token_count >= (prompt_cache_block_token_count * 2) as u32,
        "the pre-existing cache restored {warm_cached_token_count} tokens but must restore both complete prompt blocks"
    );
    assert_eq!(
        warm_generated_token_ids, cold_generated_token_ids,
        "cold and restored deterministic output must match"
    );
}

async fn run_one_token_request(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    prompt_token_ids: &[u32],
) -> (u32, Vec<u32>) {
    let generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(request_id, prompt_token_ids.to_vec(), 1)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the qualification engine should accept the request");
    let cached_token_count = generation_start.cached_token_count();
    let (generated_token_ids, _) = generate_token_ids(qwen3_5_engine, request_id, 1).await;
    assert_eq!(generated_token_ids.len(), 1);
    (cached_token_count, generated_token_ids)
}
