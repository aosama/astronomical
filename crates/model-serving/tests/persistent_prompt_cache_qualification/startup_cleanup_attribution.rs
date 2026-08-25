use std::{fs, time::Duration};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerPersistentPromptCacheLookupOutcome, WorkerPersistentPromptCacheMissReason,
};
use astronomical_model_serving::{
    InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5InferenceRequest, Qwen3_5Tokenizer,
};
use tokio::time::timeout;

use super::engine_prompt_cache::{
    generate_token_ids, load_persistent_prompt_cache_qualification_engine,
    wait_for_persistent_prompt_cache_blocks,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "loads Ornith for a Romeo and Juliet startup-cleanup attribution journey"]
async fn should_attribute_startup_invalidation_once_then_restore_the_rebuilt_cache() {
    eprintln!(
        "[prompt-cache-startup-attribution] status=start timeout_seconds={}",
        QUALIFICATION_TIMEOUT.as_secs()
    );
    timeout(
        QUALIFICATION_TIMEOUT,
        run_startup_cleanup_attribution_journey(),
    )
    .await
    .expect("the startup-cleanup qualification should finish within 115 seconds");
    eprintln!("[prompt-cache-startup-attribution] status=success");
}

async fn run_startup_cleanup_attribution_journey() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let (short_prompt_token_ids, prompt_token_ids) =
        romeo_and_juliet_prompt_token_ids(&model_directory);
    let persistent_prompt_cache_directory =
        tempfile::tempdir().expect("the qualification should create an isolated cache directory");
    let obsolete_sequence_directory = persistent_prompt_cache_directory.path().join("kv_blocks");
    fs::create_dir_all(&obsolete_sequence_directory)
        .expect("the qualification should create obsolete format storage");
    let obsolete_artifact_byte_count = 73_u64;
    fs::write(
        obsolete_sequence_directory.join(format!("{}.safetensors", "a".repeat(64))),
        vec![0_u8; obsolete_artifact_byte_count as usize],
    )
    .expect("the qualification should seed one obsolete format artifact");

    let (mut qwen3_5_engine, _, _, prompt_cache_block_token_count) =
        load_persistent_prompt_cache_qualification_engine(
            &model_directory,
            persistent_prompt_cache_directory.path(),
            4_096,
        )
        .await;
    let short_request_id = RequestId::new(41_001);
    let short_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(short_request_id, short_prompt_token_ids, 1)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the short Romeo and Juliet request should start");
    let short_diagnostics = short_generation_start
        .persistent_prompt_cache_diagnostics()
        .expect("the short request should report prompt-cache diagnostics");
    assert_eq!(
        short_diagnostics.miss_reason,
        Some(WorkerPersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache)
    );
    assert_eq!(short_diagnostics.startup_cleanup_evidence, None);
    generate_token_ids(&mut qwen3_5_engine, short_request_id, 1).await;

    let cold_request_id = RequestId::new(41_002);
    let cold_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(cold_request_id, prompt_token_ids.clone(), 1)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the cold Romeo and Juliet request should start");
    assert_eq!(cold_generation_start.cached_token_count(), 0);
    let cold_diagnostics = cold_generation_start
        .persistent_prompt_cache_diagnostics()
        .expect("the cold request should report prompt-cache diagnostics");
    assert_eq!(
        cold_diagnostics.lookup_outcome,
        WorkerPersistentPromptCacheLookupOutcome::Miss
    );
    assert_eq!(
        cold_diagnostics.miss_reason,
        Some(WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing)
    );
    let startup_cleanup_evidence = cold_diagnostics
        .startup_cleanup_evidence
        .expect("the first structural miss should carry startup cleanup evidence");
    assert_eq!(startup_cleanup_evidence.obsolete_format.artifact_count, 1);
    assert_eq!(
        startup_cleanup_evidence.obsolete_format.byte_count,
        obsolete_artifact_byte_count
    );
    assert_eq!(
        startup_cleanup_evidence
            .interrupted_transaction_recovery
            .artifact_count,
        0
    );
    assert_eq!(
        startup_cleanup_evidence.corrupt_current_format.block_count,
        0
    );
    assert_eq!(startup_cleanup_evidence.quota_eviction.block_count, 0);

    generate_token_ids(&mut qwen3_5_engine, cold_request_id, 1).await;
    wait_for_persistent_prompt_cache_blocks(&qwen3_5_engine, 1).await;

    let warm_request_id = RequestId::new(41_003);
    let warm_generation_start = qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(warm_request_id, prompt_token_ids, 1)
                .with_image_pad_token_id(248_069),
        )
        .await
        .expect("the warm Romeo and Juliet request should start");
    assert!(warm_generation_start.cached_token_count() >= prompt_cache_block_token_count as u32);
    let warm_diagnostics = warm_generation_start
        .persistent_prompt_cache_diagnostics()
        .expect("the warm request should report prompt-cache diagnostics");
    assert_eq!(
        warm_diagnostics.lookup_outcome,
        WorkerPersistentPromptCacheLookupOutcome::Hit
    );
    assert_eq!(warm_diagnostics.startup_cleanup_evidence, None);
    generate_token_ids(&mut qwen3_5_engine, warm_request_id, 1).await;
}

fn romeo_and_juliet_prompt_token_ids(model_directory: &std::path::Path) -> (Vec<u32>, Vec<u32>) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Ornith artifact should validate before prompt preparation");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Ornith tokenizer should load");
    let short_source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(200)
        .collect::<String>();
    let short_prompt_token_ids = tokenizer
        .prepare_chat(
            &romeo_and_juliet_command(41_000, short_source_excerpt, validated_artifact.model_id()),
            false,
        )
        .expect("the short Romeo and Juliet prompt should prepare")
        .input_token_ids()
        .to_vec();
    let prompt_token_ids = tokenizer
        .prepare_chat(
            &romeo_and_juliet_command(
                41_001,
                ROMEO_AND_JULIET_SOURCE.to_owned(),
                validated_artifact.model_id(),
            ),
            false,
        )
        .expect("the Romeo and Juliet prompt should prepare")
        .input_token_ids()
        .to_vec();
    (short_prompt_token_ids, prompt_token_ids)
}

fn romeo_and_juliet_command(
    request_id: u64,
    source_material: String,
    model_id: &str,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Summarize the supplied Romeo and Juliet source while preserving the central conflict and tragic outcome.\n\n{source_material}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 1,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    }
}
