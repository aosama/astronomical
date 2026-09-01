use std::io::Cursor;
use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId,
};
use astronomical_model_serving::{
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5Tokenizer,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

use super::tool_control::{
    assert_schema_valid_literary_analysis_tool_call, literary_analysis_tools, parse_one_tool_call,
};
use super::{
    RepresentativePrompt, SPECULATIVE_PREFILL_KEEP_PERCENTAGE, run_representative_generation,
};

// Exception to the repository's 120-second default test-process boundary,
// approved for these visual journeys: each leg loads the large sparse MoE target
// into wired GPU memory, prefills a visual prompt above the speculative-prefill
// floor, and generates a capped output budget, so the journey legitimately needs
// more than two minutes on machines slower than the development workstation.
const VISUAL_TOOL_ACCEPTANCE_TIMEOUT: Duration = Duration::from_secs(240);
// The visual prompts must exceed the speculative-prefill minimum prompt floor
// (8,192 tokens) so the protected journeys genuinely engage drafter scoring;
// a prompt below the floor makes every SpecPrefill measurement structurally zero.
const VISUAL_TOOL_MINIMUM_PROMPT_TOKEN_COUNT: usize = 8_704;
const VISUAL_TOOL_OUTPUT_TOKEN_COUNT: u16 = 512;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "proves an image-bearing tool call, mandatory image positions, and changed-image SSD isolation"]
async fn should_preserve_a_visual_tool_call_and_reject_prior_state_for_a_changed_image() {
    tokio::time::timeout(VISUAL_TOOL_ACCEPTANCE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let (draft_model_directory, draft_model_id) =
            crate::serving_acceptance::support::configured_speculative_prefill_draft_model(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, VISUAL_TOOL_OUTPUT_TOKEN_COUNT as u32)
            .expect("the visual tool target artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the visual tool tokenizer should load");
        let declared_tools = literary_analysis_tools();
        let baseline_visual_prompt = prepare_visual_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
            one_pixel_png(64, 96, 128),
        );
        let changed_image_visual_prompt = prepare_visual_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
            one_pixel_png(160, 32, 96),
        );
        assert_eq!(
            baseline_visual_prompt.prompt_token_ids,
            changed_image_visual_prompt.prompt_token_ids,
            "equal-dimension image changes must isolate state through ordered digests rather than token-shape differences",
        );
        let expected_mandatory_image_pad_token_count = baseline_visual_prompt.prompt_token_ids
            [baseline_visual_prompt.ordinary_target_prefill_control_span_token_count
                ..baseline_visual_prompt.prompt_token_ids.len() - 1]
            .iter()
            .filter(|prompt_token_id| {
                **prompt_token_id == baseline_visual_prompt.image_pad_token_id
            })
            .count();
        assert!(expected_mandatory_image_pad_token_count > 0);
        let shared_prompt_cache_root = tempfile::tempdir()
            .expect("the visual tool journey should create a shared SSD cache root");
        let prompt_cache_config = PersistentPromptCacheDiskStoreConfig::new(
            shared_prompt_cache_root.path().join("target"),
            shared_prompt_cache_root.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );
        let mlx_memory_limits =
            crate::common::sample_serving_acceptance_mlx_memory_limits().await;

        eprintln!("[speculative-prefill-visual-tool] status=progress phase=target_only");
        let target_only_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &baseline_visual_prompt,
            false,
            VISUAL_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_600),
            None,
            mlx_memory_limits,
        )
        .await;
        eprintln!(
            "[speculative-prefill-visual-tool] status=progress phase=protected_speculative_prefill"
        );
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &baseline_visual_prompt,
            true,
            VISUAL_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_601),
            Some(prompt_cache_config.clone()),
            mlx_memory_limits,
        )
        .await;
        let target_only_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &target_only_measurement.generated_token_ids,
        );
        let speculative_prefill_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &speculative_prefill_measurement.generated_token_ids,
        );
        assert_schema_valid_literary_analysis_tool_call(&target_only_tool_call);
        assert_schema_valid_literary_analysis_tool_call(&speculative_prefill_tool_call);
        assert_eq!(
            speculative_prefill_tool_call.function_name,
            target_only_tool_call.function_name,
        );
        assert_eq!(
            speculative_prefill_measurement.speculative_prefill_mandatory_visual_token_count,
            expected_mandatory_image_pad_token_count as u64,
        );

        eprintln!(
            "[speculative-prefill-visual-tool] status=progress phase=changed_image_digest_isolation"
        );
        let changed_image_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &changed_image_visual_prompt,
            true,
            1,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_602),
            Some(prompt_cache_config),
            mlx_memory_limits,
        )
        .await;
        assert_eq!(
            changed_image_measurement
                .speculative_prefill_target_persistent_state_restored_token_count,
            0,
            "a changed image must not restore prior selection-bound target state",
        );
        assert_eq!(
            changed_image_measurement
                .speculative_prefill_draft_persistent_prefix_restored_token_count,
            0,
            "a changed image must not restore prior dense drafter state",
        );
        assert!(
            changed_image_measurement.speculative_prefill_draft_scored_suffix_token_count > 0
        );
        eprintln!("[speculative-prefill-visual-tool] status=success");
    })
    .await
    .expect("the visual tool acceptance should finish within the acceptance timeout");
}

#[tokio::test]
#[ignore = "proves a visual prompt serves without fallback and the compatible drafter scores it"]
async fn should_score_a_visual_prompt_with_the_compatible_drafter() {
    tokio::time::timeout(VISUAL_TOOL_ACCEPTANCE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let (text_only_draft_model_directory, text_only_draft_model_id) =
            configured_compatible_draft(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, 1)
            .expect("the target-only visual target artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the target-only visual tokenizer should load");
        let declared_tools = literary_analysis_tools();
        let visual_prompt = prepare_visual_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
            one_pixel_png(64, 96, 128),
        );
        let shared_prompt_cache_root = tempfile::tempdir()
            .expect("the target-only visual journey should create an SSD cache root");
        let prompt_cache_config = PersistentPromptCacheDiskStoreConfig::new(
            shared_prompt_cache_root.path().join("target"),
            shared_prompt_cache_root.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );
        let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;

        let validated_draft_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&text_only_draft_model_directory, 1)
            .expect("the text-only draft artifact should validate");
        let draft_consumes_processed_images = validated_draft_artifact.supports_image_input();
        let target_only_visual_measurement = run_representative_generation(
            &target_model_directory,
            &text_only_draft_model_directory,
            &text_only_draft_model_id,
            &visual_prompt,
            true,
            16,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_603),
            Some(prompt_cache_config),
            mlx_memory_limits,
        )
        .await;
        if draft_consumes_processed_images {
            assert!(
                target_only_visual_measurement.speculative_prefill_selected_token_count > 0,
                "a vision-capable drafter should score a visual prompt"
            );
        } else {
            assert_eq!(
                target_only_visual_measurement.speculative_prefill_selected_token_count,
                0,
            );
            assert_eq!(
                target_only_visual_measurement.speculative_prefill_draft_scoring_elapsed_seconds,
                0.0,
            );
        }
        assert_eq!(
            target_only_visual_measurement.speculative_prefill_fallback_count,
            0
        );
        if !draft_consumes_processed_images {
            assert_eq!(
                target_only_visual_measurement
                    .speculative_prefill_target_persistent_state_write_count,
                0,
            );
        }
        eprintln!(
            "[speculative-prefill-visual-tool] status=success journey=compatible_drafter_target_only"
        );
    })
    .await
    .expect("the compatible-drafter visual journey should finish within the acceptance timeout");
}

fn configured_compatible_draft(
    _target_model_directory: &std::path::Path,
) -> (std::path::PathBuf, String) {
    let draft_model_id = crate::common::small_dense_model_id().to_owned();
    let draft_model_directory =
        crate::common::configured_installed_model_directory_by_id(&draft_model_id);
    (draft_model_directory, draft_model_id)
}

fn prepare_visual_tool_prompt(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
    image_input: ChatImageInput,
) -> RepresentativePrompt {
    let mut source_material = String::new();
    loop {
        if !source_material.is_empty() {
            source_material.push_str("\n\n");
        }
        source_material.push_str(ROMEO_AND_JULIET_SOURCE);
        let prepared_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_599),
                    model: target_model_id.to_owned(),
                    messages: vec![
                        ChatMessage::System {
                            content: "Use the image and source evidence, then call the declared tool with every required field.".to_owned(),
                        },
                        ChatMessage::User {
                            content: format!(
                                "Call record_literary_analysis now. Use the image and Romeo and Juliet source material to identify the central conflict and classify the outcome as tragic.\n\n{source_material}"
                            ),
                            images: vec![image_input.clone()],
                        },
                    ],
                    tools: declared_tools.to_vec(),
                    tool_choice: ChatToolChoice::Auto,
                    settings: ChatGenerationSettings {
                        max_output_tokens: VISUAL_TOOL_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        // A short thinking budget keeps the reasoning channel from
                        // consuming the output budget before the tool call is emitted.
                        thinking_budget: Some(64),
                    },
                    qwen_thinking_channel_seed: None,
                },
                false,
            )
            .expect("the visual tool prompt should prepare");
        if prepared_request.input_token_ids().len() >= VISUAL_TOOL_MINIMUM_PROMPT_TOKEN_COUNT {
            return RepresentativePrompt {
                prompt_token_ids: prepared_request.input_token_ids().to_vec(),
                image_pad_token_id: tokenizer.image_pad_token_id(),
                processed_visual_images: prepared_request.processed_visual_images().to_vec(),
                ordinary_target_prefill_control_span_token_count: prepared_request
                    .ordinary_target_prefill_control_span_token_count(),
                sampling_temperature_thousandths: 1_000,
                sampling_top_p_thousandths: 1_000,
                sampling_seed: None,
            };
        }
    }
}

fn one_pixel_png(red: u8, green: u8, blue: u8) -> ChatImageInput {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([red, green, blue]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the visual tool image should encode");
    ChatImageInput {
        mime_type: "image/png".to_owned(),
        decoded_bytes: encoded_image_bytes.into_inner(),
    }
}
