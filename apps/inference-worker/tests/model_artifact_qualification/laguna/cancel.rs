//! Direct-engine cancellation leaves loaded Laguna state reusable.

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, MlxInferenceExecution, initialize_laguna_execution,
};

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, compact_romeo_and_juliet_source, resolve_reference_model_directory,
};
use super::generate::resolve_machine_mlx_memory_limits;

#[test]
#[ignore = "loads reference Laguna XS and proves cancel leaves the engine reusable"]
fn should_cancel_a_laguna_generate_and_remain_reusable() {
    let model_directory = resolve_reference_model_directory();
    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        resolve_machine_mlx_memory_limits();
    let (generation_processor, mut execution) = initialize_laguna_execution(
        &model_directory,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        false,
    )
    .expect("Laguna XS startup should construct execution");
    execution.load().expect("Laguna XS weights should load");
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(106),
        model: LAGUNA_XS_PUBLIC_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Use the supplied Romeo and Juliet source. Name the households.\n\n{}",
                compact_romeo_and_juliet_source()
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 32,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(106),
            thinking_budget: Some(0),
        },
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the compact Romeo prompt should prepare");
    execution
        .start_generation(prepared_generation.into_inference_request())
        .expect("Laguna prompt processing should start");
    let first_prefill_boundary = execution
        .decode_next_token(chat_command.request_id)
        .expect("Laguna should complete one bounded prompt chunk");
    assert!(matches!(
        first_prefill_boundary,
        GeneratedToken::PrefillProgress { .. }
    ));
    // Returning progress before cancellation proves one engine advance no longer
    // owns the complete prompt-processing loop.
    execution
        .cancel_generation(chat_command.request_id)
        .expect("cancel should succeed between Laguna prompt chunks");
    let second_prepared = generation_processor
        .prepare_chat(&chat_command)
        .expect("the compact Romeo prompt should prepare again");
    execution
        .start_generation(second_prepared.into_inference_request())
        .expect("the loaded Laguna engine should accept a new generation after cancel");
    assert!(matches!(
        execution
            .decode_next_token(chat_command.request_id)
            .expect("the reused engine should process another bounded prompt chunk"),
        GeneratedToken::PrefillProgress { .. }
    ));
    execution
        .cancel_generation(chat_command.request_id)
        .expect("the reused Laguna engine should cancel again");
}
