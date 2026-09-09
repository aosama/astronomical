use astronomical_ipc_protocol::{
    ChatGenerationOutput, ChatMessage, ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, K2HorizonMoVAInferenceRequest, K2HorizonMoVAPromptRenderer,
    K2HorizonMoVARequestOutput, K2HorizonMoVAServingSettings, K2HorizonMoVATokenizer,
    MlxInferenceExecution, initialize_k2_horizon_mova_execution,
    initialize_k2_horizon_mova_execution_with_serving_settings,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "requires model_directories to discover a stacked affine K2 Horizon MoVA artifact"]
async fn should_generate_tokens_from_romeo_and_juliet_on_gpu() {
    let outputs = generate_k2_outputs(
        "Name the two households in the supplied Romeo and Juliet source.",
        &[],
        &ChatToolChoice::None,
        48,
    )
    .await;
    let visible = flatten_outputs(&outputs);
    eprintln!("[k2-horizon-mova] decoded_output={visible:?}");
    assert!(
        visible.chars().any(|character| character.is_alphabetic()),
        "K2 Horizon MoVA must decode alphabetic chat text, got {visible:?}"
    );
}

#[tokio::test]
#[ignore = "requires model_directories to discover a stacked affine K2 Horizon MoVA artifact"]
async fn should_honor_ifm_tool_catalog_on_gpu() {
    let outputs = generate_k2_outputs(
        "Call get_scene with title Romeo and Juliet. Do not answer in prose.",
        &[ChatToolDefinition {
            name: "get_scene".to_owned(),
            description: Some("Return a scene from the play".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}"#
                    .to_owned(),
        }],
        &ChatToolChoice::Auto,
        96,
    )
    .await;
    let visible = flatten_outputs(&outputs);
    eprintln!("[k2-horizon-mova] tool_journey_output={visible:?}");
    assert!(
        outputs.iter().any(|output| {
            matches!(
                output,
                ChatGenerationOutput::ToolCall { function_name, arguments_json, .. }
                    if function_name == "get_scene" && arguments_json.contains("Romeo")
            )
        }),
        "K2 Horizon MoVA must honor get_scene as a tool call, got {outputs:?}"
    );
}

async fn generate_k2_outputs(
    user_instruction: &str,
    tools: &[ChatToolDefinition],
    tool_choice: &ChatToolChoice,
    max_output_tokens: u32,
) -> Vec<ChatGenerationOutput> {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = configured_k2_horizon_mova_model_directory();
    eprintln!(
        "[k2-horizon-mova] loading stacked affine artifact leaf {}",
        model_directory
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_owned())
    );
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let (_processor, mut execution) = initialize_k2_horizon_mova_execution(
        &model_directory,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        true,
    )
    .expect("configured K2 Horizon MoVA artifact should start");
    execution
        .load()
        .expect("K2 Horizon MoVA weights should load on GPU");
    eprintln!("[k2-horizon-mova] load complete");
    let prompt_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(280)
        .collect::<String>();
    let renderer = K2HorizonMoVAPromptRenderer::new();
    let prompt = renderer.render(
        &[ChatMessage::User {
            content: format!("{user_instruction}\n\n{prompt_excerpt}"),
            images: Vec::new(),
        }],
        tools,
        tool_choice,
    );
    let tokenizer = K2HorizonMoVATokenizer::from_json_bytes(
        &std::fs::read(model_directory.join("tokenizer.json"))
            .expect("tokenizer.json should be readable"),
        &astronomical_model_serving::K2HorizonMoVAConfig::from_json_bytes(
            &std::fs::read(model_directory.join("config.json"))
                .expect("config.json should be readable"),
        )
        .expect("family config should parse"),
    )
    .expect("tokenizer should load");
    let prompt_token_ids = tokenizer
        .encode_prompt(&prompt)
        .expect("Romeo and Juliet prompt should encode");
    eprintln!(
        "[k2-horizon-mova] prompt_token_count={}",
        prompt_token_ids.len()
    );
    let inference_request = K2HorizonMoVAInferenceRequest::new(
        prompt_token_ids,
        max_output_tokens,
        1_000,
        950,
        Some(1),
    );
    execution
        .start_generation(inference_request)
        .expect("prefill should start");
    eprintln!("[k2-horizon-mova] prefill complete");
    let mut request_output = K2HorizonMoVARequestOutput::new_with_declared_tool_names(
        &tokenizer,
        tools.iter().map(|tool| tool.name.clone()).collect(),
    );
    let mut outputs = Vec::new();
    let mut generated_token_count = 0_u32;
    for decode_step in 0..(max_output_tokens.saturating_add(32)) {
        match execution
            .decode_next_token(RequestId::new(u64::from(decode_step) + 1))
            .expect("decode should produce a token, prefill progress, or end")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => {
                eprintln!(
                    "[k2-horizon-mova] prefill_processed_token_count={processed_token_count}"
                );
            }
            GeneratedToken::TokenId { token_id, .. } => {
                generated_token_count += 1;
                outputs.extend(
                    request_output
                        .push_token(token_id)
                        .expect("token should decode"),
                );
                if generated_token_count >= max_output_tokens {
                    break;
                }
            }
            GeneratedToken::EndOfSequence => break,
            other => panic!("unexpected generation event: {other:?}"),
        }
    }
    outputs.extend(request_output.finish());
    outputs
}

fn flatten_outputs(outputs: &[ChatGenerationOutput]) -> String {
    let mut flattened = String::new();
    for output in outputs {
        match output {
            ChatGenerationOutput::Reasoning { text } | ChatGenerationOutput::Text { text } => {
                flattened.push_str(text);
            }
            ChatGenerationOutput::ToolCall {
                function_name,
                arguments_json,
                ..
            } => {
                flattened.push_str(function_name);
                flattened.push_str(arguments_json);
            }
        }
    }
    flattened
}

fn configured_k2_horizon_mova_model_directory() -> std::path::PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::k2_horizon_mova_model_id(),
    )
}

#[tokio::test]
#[ignore = "measures full-model K2 decode tok/s with the fused expert kernels on or off"]
async fn should_measure_fused_expert_decode_tps() {
    let fused_expert_decode_enabled = std::env::var("K2_FUSED_MOE")
        .map(|value| value != "0")
        .unwrap_or(true);
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = configured_k2_horizon_mova_model_directory();
    let memory_limits = crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    let serving_settings = K2HorizonMoVAServingSettings::default_fixed();
    let serving_settings = K2HorizonMoVAServingSettings {
        fused_expert_decode_enabled,
        ..serving_settings
    };
    let (_processor, mut execution) = initialize_k2_horizon_mova_execution_with_serving_settings(
        &model_directory,
        memory_limits.active_memory_limit_bytes(),
        memory_limits.allocator_cache_memory_limit_bytes(),
        false,
        serving_settings,
    )
    .expect("configured K2 artifact should start");
    execution
        .load()
        .expect("K2 Horizon MoVA weights should load on GPU");

    // ~3k prompt tokens so decode runs over a meaningful KV context.
    let prompt_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(5_500)
        .collect::<String>();
    let renderer = K2HorizonMoVAPromptRenderer::new();
    let prompt = renderer.render(
        &[astronomical_ipc_protocol::ChatMessage::User {
            content: format!(
                "Name the two households in the supplied Romeo and Juliet source.\n\n{prompt_excerpt}"
            ),
            images: Vec::new(),
        }],
        &[],
        &astronomical_ipc_protocol::ChatToolChoice::None,
    );
    let tokenizer = K2HorizonMoVATokenizer::from_json_bytes(
        &std::fs::read(model_directory.join("tokenizer.json")).expect("tokenizer"),
        &astronomical_model_serving::K2HorizonMoVAConfig::from_json_bytes(
            &std::fs::read(model_directory.join("config.json")).expect("config"),
        )
        .expect("config"),
    )
    .expect("tokenizer");
    let prompt_token_ids = tokenizer.encode_prompt(&prompt).expect("encode");
    eprintln!(
        "[k2-fused-bench] fused={fused_expert_decode_enabled} prompt_tokens={}",
        prompt_token_ids.len()
    );
    let inference_request =
        K2HorizonMoVAInferenceRequest::new(prompt_token_ids, 192, 1_000, 950, Some(1));
    execution
        .start_generation(inference_request)
        .expect("prefill should start");
    let mut generated_token_count = 0_u32;
    let decode_start = std::time::Instant::now();
    for decode_step in 0..224_u32 {
        match execution
            .decode_next_token(astronomical_ipc_protocol::RequestId::new(
                u64::from(decode_step) + 1,
            ))
            .expect("decode step")
        {
            GeneratedToken::PrefillProgress { .. } => continue,
            GeneratedToken::TokenId { .. } => generated_token_count += 1,
            GeneratedToken::EndOfSequence => break,
            other => panic!("unexpected generation event: {other:?}"),
        }
        if generated_token_count >= 192 {
            break;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let tokens_per_second = f64::from(generated_token_count) / decode_seconds;
    eprintln!(
        "[k2-fused-bench] fused={fused_expert_decode_enabled} tokens={generated_token_count} seconds={decode_seconds:.2} tok_per_second={tokens:.2}",
        tokens = tokens_per_second,
    );
    assert!(generated_token_count > 0, "decode must produce tokens");
}
