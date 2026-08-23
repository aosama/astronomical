//! Real Laguna journey proving deterministic resident-to-paged transition under a live ceiling.

use std::process::Stdio;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    ExpertMemoryMode, GeneratedToken, LagunaArtifactValidator, MlxInferenceExecution,
    initialize_laguna_execution,
};
use astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

const LIVE_TRANSITION_CHILD_KEY: &str = "ASTRONOMICAL_LAGUNA_MEMORY_ACCEPTANCE_CHILD";
const CONSTRAINED_STARTUP_CHILD_KEY: &str =
    "ASTRONOMICAL_LAGUNA_CONSTRAINED_STARTUP_ACCEPTANCE_CHILD";
const LIVE_TRANSITION_TEST_NAME: &str = "model_ssd_streaming::laguna_paging_journey::should_preserve_laguna_output_across_resident_to_model_ssd_streaming_transition";
const CONSTRAINED_STARTUP_TEST_NAME: &str = "model_ssd_streaming::laguna_paging_journey::should_stream_laguna_from_ssd_when_the_ceiling_is_below_weight_files";
const LAGUNA_XS_PUBLIC_MODEL_ID: &str = "Laguna-XS-2.1-oQ8e";
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test]
#[ignore = "loads configured Laguna XS and proves centralized live-ceiling demotion parity"]
async fn should_preserve_laguna_output_across_resident_to_model_ssd_streaming_transition() {
    if std::env::var_os(LIVE_TRANSITION_CHILD_KEY).is_some() {
        run_real_laguna_memory_journey();
        return;
    }
    run_isolated_acceptance(LIVE_TRANSITION_TEST_NAME, LIVE_TRANSITION_CHILD_KEY).await;
}

#[tokio::test]
#[ignore = "loads configured Laguna XS below its weight-file payload and generates through paging"]
async fn should_stream_laguna_from_ssd_when_the_ceiling_is_below_weight_files() {
    if std::env::var_os(CONSTRAINED_STARTUP_CHILD_KEY).is_some() {
        run_constrained_startup_journey();
        return;
    }
    run_isolated_acceptance(CONSTRAINED_STARTUP_TEST_NAME, CONSTRAINED_STARTUP_CHILD_KEY).await;
}

async fn run_isolated_acceptance(test_name: &str, child_environment_key: &str) {
    let test_executable = std::env::current_exe().expect("the acceptance test binary should exist");
    let mut child = Command::new(test_executable)
        .args(["--ignored", "--exact", test_name, "--nocapture"])
        .env(child_environment_key, "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the isolated Laguna memory journey should start");
    let child_status = timeout(Duration::from_secs(115), child.wait())
        .await
        .expect("the Laguna memory journey must finish within 115 seconds")
        .expect("the isolated Laguna memory journey status should be readable");
    assert!(
        child_status.success(),
        "the isolated Laguna memory journey failed"
    );
}

fn run_constrained_startup_journey() {
    let model_directory =
        crate::common::configured_model_artifact_directory_by_id(LAGUNA_XS_PUBLIC_MODEL_ID);
    let weight_file_payload_bytes = LagunaArtifactValidator::new()
        .validate(&model_directory)
        .expect("the configured Laguna artifact should validate")
        .total_shard_file_bytes();
    let constrained_memory_ceiling_bytes = usize::try_from(weight_file_payload_bytes / 2)
        .expect("half the Laguna weight payload should fit the platform integer range");
    let machine_memory_ceiling_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("the machine GPU working-set recommendation should be readable");
    let constrained_memory_ceiling_bytes =
        constrained_memory_ceiling_bytes.min(machine_memory_ceiling_bytes);
    assert!(
        u64::try_from(constrained_memory_ceiling_bytes).unwrap_or(u64::MAX)
            < weight_file_payload_bytes,
        "the constrained startup ceiling must remain below the weight files on disk"
    );
    eprintln!(
        "[laguna-constrained-startup] status=progress phase=load weight_file_bytes={weight_file_payload_bytes} ceiling_bytes={constrained_memory_ceiling_bytes}"
    );
    let (generation_processor, mut execution) = initialize_laguna_execution(
        &model_directory,
        constrained_memory_ceiling_bytes,
        constrained_memory_ceiling_bytes,
        true,
    )
    .expect("constrained Laguna XS startup should prepare");
    let load_result = execution
        .load()
        .expect("Laguna XS should load below its on-disk weight payload");
    assert_ne!(
        load_result.expert_memory_mode(),
        Some(ExpertMemoryMode::Resident),
        "a ceiling below the weight payload must select bounded expert paging"
    );
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(4_000)
        .collect::<String>();
    let (generated_token_ids, finalization) = generate_token_ids(
        &generation_processor,
        &mut execution,
        RequestId::new(103_003),
        &source_excerpt,
    );
    assert_ne!(
        finalization.expert_memory_mode(),
        Some(ExpertMemoryMode::Resident),
        "constrained generation must not claim complete residency"
    );
    eprintln!(
        "[laguna-constrained-startup] status=success weight_file_bytes={weight_file_payload_bytes} ceiling_bytes={constrained_memory_ceiling_bytes} generated_tokens={} final_mode={:?}",
        generated_token_ids.len(),
        finalization.expert_memory_mode()
    );
}

fn run_real_laguna_memory_journey() {
    let model_directory =
        crate::common::configured_model_artifact_directory_by_id(LAGUNA_XS_PUBLIC_MODEL_ID);
    let machine_memory_ceiling_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("the machine GPU working-set recommendation should be readable");
    let startup_memory_ceiling_bytes = machine_memory_ceiling_bytes.saturating_sub(1);
    assert!(
        startup_memory_ceiling_bytes > 0,
        "the Laguna acceptance journey needs a positive startup ceiling"
    );
    eprintln!(
        "[laguna-memory-acceptance] status=progress phase=load ceiling_bytes={startup_memory_ceiling_bytes}"
    );
    let (generation_processor, mut execution) = initialize_laguna_execution(
        &model_directory,
        startup_memory_ceiling_bytes,
        startup_memory_ceiling_bytes,
        true,
    )
    .expect("configured Laguna XS startup should prepare");
    let load_result = execution.load().expect("configured Laguna XS should load");
    assert_eq!(
        load_result.expert_memory_mode(),
        Some(ExpertMemoryMode::Resident),
        "this acceptance cell requires fitting complete Laguna XS residency"
    );
    let minimum_mlx_memory_ceiling_bytes = load_result.minimum_mlx_memory_ceiling_bytes();
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(4_000)
        .collect::<String>();
    let (resident_output, _) = generate_token_ids(
        &generation_processor,
        &mut execution,
        RequestId::new(103_001),
        &source_excerpt,
    );

    eprintln!(
        "[laguna-memory-acceptance] status=progress phase=lower ceiling_bytes={minimum_mlx_memory_ceiling_bytes}"
    );
    let lowered_adjustment = execution
        .update_mlx_memory_limit(minimum_mlx_memory_ceiling_bytes)
        .expect("the advertised minimum Laguna ceiling should remain executable");
    assert_ne!(
        lowered_adjustment.expert_memory_mode(),
        ExpertMemoryMode::Resident,
        "lowering to the safe minimum must release indivisible native routed experts"
    );
    let (paged_output, _) = generate_token_ids(
        &generation_processor,
        &mut execution,
        RequestId::new(103_002),
        &source_excerpt,
    );
    assert_eq!(
        paged_output, resident_output,
        "resident and constrained paging must preserve deterministic output tokens"
    );

    let raised_adjustment = execution
        .update_mlx_memory_limit(machine_memory_ceiling_bytes as u64)
        .expect("raising the Laguna ceiling should succeed without eager reads");
    assert_eq!(
        raised_adjustment.allocator_cache_memory_limit_bytes(),
        startup_memory_ceiling_bytes as u64,
        "raising the active ceiling must retain Laguna's startup allocator-cache cap"
    );
    assert_ne!(
        raised_adjustment.expert_memory_mode(),
        ExpertMemoryMode::Resident,
        "raising capacity alone must not perform eager expert source reads"
    );
    eprintln!(
        "[laguna-memory-acceptance] status=success generated_tokens={} lowered_mode={:?}",
        paged_output.len(),
        lowered_adjustment.expert_memory_mode()
    );
}

fn generate_token_ids(
    generation_processor: &astronomical_model_serving::LagunaGenerationProcessor,
    execution: &mut astronomical_model_serving::LagunaInferenceExecution,
    request_id: RequestId,
    source_excerpt: &str,
) -> (Vec<u32>, astronomical_model_serving::GenerationFinalization) {
    let command = ChatGenerationCommand {
        request_id,
        model: LAGUNA_XS_PUBLIC_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Use this Romeo and Juliet source as the only evidence. Name the households and tragic outcome in two short sentences.\n\n{source_excerpt}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 4,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(103),
            thinking_budget: Some(0),
        },
    };
    let prepared_generation = generation_processor
        .prepare_chat(&command)
        .expect("the Romeo and Juliet acceptance prompt should prepare");
    execution
        .start_generation(prepared_generation.into_inference_request())
        .expect("the Laguna acceptance generation should start");
    let mut generated_token_ids = Vec::new();
    for advance_index in 0..128 {
        match execution
            .decode_next_token(request_id)
            .expect("the Laguna acceptance generation should advance")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => eprintln!(
                "[laguna-memory-acceptance] status=progress phase=prefill advance={advance_index} processed_tokens={processed_token_count}"
            ),
            GeneratedToken::TokenId { token_id, .. } => generated_token_ids.push(token_id),
            GeneratedToken::EndOfSequence => break,
            other => panic!("Laguna emitted an unexpected acceptance boundary: {other:?}"),
        }
        if generated_token_ids.len() == 4 {
            break;
        }
    }
    let finalization = execution
        .cancel_generation(request_id)
        .expect("the Laguna acceptance generation should finalize cleanly");
    assert!(
        !generated_token_ids.is_empty(),
        "Laguna must generate at least one token"
    );
    (generated_token_ids, finalization)
}
