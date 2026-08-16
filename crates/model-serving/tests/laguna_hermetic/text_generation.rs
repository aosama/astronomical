use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatImageInput, ChatMessage, ChatToolChoice,
};
use astronomical_model_serving::{
    LagunaGenerationProcessor, LagunaOutputEvent, LagunaOutputParserError, LagunaPreparationError,
    LagunaPromptRenderer, LagunaTokenizer, ModelGenerationProcessor,
};
use serde_json::json;

use super::text_support::{
    ASSISTANT_END_TOKEN_ID, DEFAULT_SYSTEM_MESSAGE, GENERATION_ONLY_EOS_TOKEN_ID,
    ROMEO_AND_JULIET_SOURCE, SYNTHETIC_LAGUNA_MODEL_ID, SyntheticLagunaTextArtifact,
    romeo_and_juliet_command, template_with_defaults,
};

#[test]
fn should_prepare_and_parse_the_complete_romeo_and_juliet_chat_journey() {
    let text_descriptor = SyntheticLagunaTextArtifact::small_included().normalize();
    let processor = LagunaGenerationProcessor::new(SYNTHETIC_LAGUNA_MODEL_ID, text_descriptor)
        .expect("the validated Laguna text descriptor should construct a processor");
    let chat_command = romeo_and_juliet_command(9_801, None);

    let prepared_generation = processor
        .prepare_chat(&chat_command)
        .expect("the text-only Romeo and Juliet journey should prepare");

    assert!(!prepared_generation.prompt_token_ids().is_empty());
    assert!(
        prepared_generation
            .rendered_prompt()
            .contains(ROMEO_AND_JULIET_SOURCE)
    );
    assert!(prepared_generation.thinking_enabled());
    assert!(prepared_generation.generation_starts_in_reasoning());
    assert_eq!(prepared_generation.sampler_config().top_k(), Some(20));
    assert_eq!(
        prepared_generation
            .sampler_config()
            .repetition_penalty_thousandths(),
        1_050
    );
    assert!(prepared_generation.is_end_token(ASSISTANT_END_TOKEN_ID));
    assert!(prepared_generation.is_end_token(GENERATION_ONLY_EOS_TOKEN_ID));

    let mut output_parser = prepared_generation
        .new_output_parser()
        .expect("the validated poolside_v1 contract should construct its parser");
    let output_events = output_parser
        .push_fragment(
            "Inspect both characters</think>They are central.\
             <tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>\
             <tool_call>summarize_scene<arg_key>scene</arg_key><arg_value>balcony</arg_value></tool_call>",
        )
        .expect("reasoning, text, and multiple Poolside tool calls should stream");

    assert_eq!(
        output_events,
        vec![
            LagunaOutputEvent::ReasoningDelta("Inspect both characters".to_owned()),
            LagunaOutputEvent::TextDelta("They are central.".to_owned()),
            LagunaOutputEvent::ToolCall {
                index: 0,
                function_name: "find_character".to_owned(),
                arguments_json: r#"{"name":"Romeo"}"#.to_owned(),
            },
            LagunaOutputEvent::ToolCall {
                index: 1,
                function_name: "summarize_scene".to_owned(),
                arguments_json: r#"{"scene":"balcony"}"#.to_owned(),
            },
        ]
    );
    assert!(
        output_parser
            .finish()
            .expect("the complete user journey output should finish cleanly")
            .is_empty()
    );
}

#[test]
fn should_render_then_tokenize_with_laguna_owned_public_components() {
    let text_descriptor = SyntheticLagunaTextArtifact::extra_small_inline().normalize();
    let chat_command = romeo_and_juliet_command(9_802, Some(0));
    let prompt_renderer = LagunaPromptRenderer::new(&text_descriptor);
    let rendered_prompt = prompt_renderer
        .render(&chat_command.messages, &chat_command.tools, false)
        .expect("the validated supported Poolside template should render");
    let tokenizer = LagunaTokenizer::from_descriptor(&text_descriptor)
        .expect("the validated synthetic tokenizer should construct");

    let prompt_token_ids = tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the rendered Romeo and Juliet prompt should tokenize");

    assert!(!prompt_token_ids.is_empty());
    assert!(rendered_prompt.contains(ROMEO_AND_JULIET_SOURCE));
    assert!(rendered_prompt.ends_with("<assistant></think>"));
}

#[test]
fn should_apply_thinking_precedence_as_request_then_generation_config_then_template() {
    // Generation config true overrides the inline template's false default.
    let generation_true_processor = processor(SyntheticLagunaTextArtifact::extra_small_inline());
    assert!(
        generation_true_processor
            .prepare_chat(&romeo_and_juliet_command(9_803, None))
            .expect("generation-config thinking should prepare")
            .thinking_enabled()
    );

    // An explicit zero request budget disables thinking above every artifact default.
    assert!(
        !generation_true_processor
            .prepare_chat(&romeo_and_juliet_command(9_804, Some(0)))
            .expect("an explicit thinking-off request should prepare")
            .thinking_enabled()
    );

    let mut generation_false_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    generation_false_artifact.tokenizer_config["chat_template"] =
        json!(template_with_defaults(true, false));
    generation_false_artifact
        .generation_config
        .as_mut()
        .expect("the fixture has generation configuration")["default_chat_template_kwargs"]["enable_thinking"] =
        json!(false);
    let generation_false_processor = processor(generation_false_artifact);

    // Generation config false overrides the template's true default.
    assert!(
        !generation_false_processor
            .prepare_chat(&romeo_and_juliet_command(9_805, None))
            .expect("generation-config thinking-off should prepare")
            .thinking_enabled()
    );

    // A positive explicit request budget has the highest precedence and remains available.
    let request_override = generation_false_processor
        .prepare_chat(&romeo_and_juliet_command(9_806, Some(256)))
        .expect("an explicit thinking-on request should prepare");
    assert!(request_override.thinking_enabled());
    assert_eq!(request_override.thinking_budget(), Some(256));

    let mut template_default_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    template_default_artifact.tokenizer_config["chat_template"] =
        json!(template_with_defaults(true, false));
    template_default_artifact
        .generation_config
        .as_mut()
        .and_then(|generation_config| {
            generation_config["default_chat_template_kwargs"].as_object_mut()
        })
        .expect("template kwargs should be an object")
        .remove("enable_thinking");
    assert!(
        processor(template_default_artifact)
            .prepare_chat(&romeo_and_juliet_command(9_807, None))
            .expect("the template-default thinking request should prepare")
            .thinking_enabled()
    );
}

#[test]
fn should_resolve_generation_sampling_and_explicit_request_overrides() {
    let processor = processor(SyntheticLagunaTextArtifact::small_included());
    let mut chat_command = romeo_and_juliet_command(9_808, None);
    chat_command.settings.temperature_thousandths = Some(350);
    chat_command.settings.top_p_thousandths = Some(825);

    let prepared_generation = processor
        .prepare_chat(&chat_command)
        .expect("explicit request sampling should prepare");
    let resolved_sampler = prepared_generation.sampler_config();

    assert!(resolved_sampler.uses_sampling());
    assert_eq!(resolved_sampler.temperature_thousandths(), 350);
    assert_eq!(resolved_sampler.top_p_thousandths(), 825);
    assert_eq!(resolved_sampler.top_k(), Some(20));
    assert_eq!(resolved_sampler.repetition_penalty_thousandths(), 1_050);
    assert_eq!(resolved_sampler.seed(), Some(98));
}

#[test]
fn should_preserve_generation_config_greedy_policy_without_request_overrides() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact
        .generation_config
        .as_mut()
        .expect("the fixture has generation configuration")["do_sample"] = json!(false);

    let prepared_generation = processor(text_artifact)
        .prepare_chat(&romeo_and_juliet_command(9_809, None))
        .expect("the greedy generation policy should prepare");

    assert!(!prepared_generation.sampler_config().uses_sampling());
}

#[test]
fn should_apply_explicit_request_sampling_precedence_over_a_greedy_artifact_default() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact
        .generation_config
        .as_mut()
        .expect("the fixture has generation configuration")["do_sample"] = json!(false);
    let processor = processor(text_artifact);
    let mut sampling_command = romeo_and_juliet_command(9_820, None);
    sampling_command.settings.temperature_thousandths = Some(350);
    sampling_command.settings.top_p_thousandths = Some(825);

    let sampled_generation = processor
        .prepare_chat(&sampling_command)
        .expect("explicit nonzero request sampling should override the artifact default");

    assert!(sampled_generation.sampler_config().uses_sampling());
    assert_eq!(
        sampled_generation
            .sampler_config()
            .temperature_thousandths(),
        350
    );
    assert_eq!(sampled_generation.sampler_config().top_p_thousandths(), 825);

    let mut greedy_command = romeo_and_juliet_command(9_821, None);
    greedy_command.settings.temperature_thousandths = Some(0);
    greedy_command.settings.top_p_thousandths = Some(825);
    let greedy_generation = processor
        .prepare_chat(&greedy_command)
        .expect("an explicit zero request temperature should remain greedy");

    assert!(!greedy_generation.sampler_config().uses_sampling());
    assert_eq!(
        greedy_generation.sampler_config().temperature_thousandths(),
        0
    );
    assert_eq!(greedy_generation.sampler_config().top_p_thousandths(), 825);
}

#[test]
fn should_hide_tools_and_reject_generated_calls_when_tool_choice_is_none() {
    let processor = processor(SyntheticLagunaTextArtifact::extra_small_inline());
    let mut chat_command = romeo_and_juliet_command(9_822, Some(0));
    chat_command.tool_choice = ChatToolChoice::None;
    assert_eq!(chat_command.tools.len(), 2);

    let prepared_generation = processor
        .prepare_chat(&chat_command)
        .expect("tool_choice none should prepare while retaining typed command definitions");

    // Request policy removes tools from both model-visible prompting and request-local parsing.
    assert!(
        !prepared_generation
            .rendered_prompt()
            .contains("<available_tools>")
    );
    assert!(
        !prepared_generation
            .rendered_prompt()
            .contains("find_character")
    );
    assert!(
        !prepared_generation
            .rendered_prompt()
            .contains("summarize_scene")
    );

    let mut output_parser = prepared_generation
        .new_output_parser()
        .expect("the no-tools request should construct an empty request-local parser");
    let parser_error = output_parser
        .push_fragment(
            "<tool_call>find_character<arg_key>name</arg_key><arg_value>Romeo</arg_value></tool_call>",
        )
        .expect_err("a generated call hidden by tool_choice none must be undeclared");

    assert!(matches!(
        parser_error,
        LagunaOutputParserError::UndeclaredFunction { function_name }
            if function_name == "find_character"
    ));
}

#[test]
fn should_render_default_explicit_and_intentionally_empty_system_messages() {
    let processor = processor(SyntheticLagunaTextArtifact::extra_small_inline());

    let mut default_system_command = romeo_and_juliet_command(9_810, Some(0));
    default_system_command.tools.clear();
    default_system_command.tool_choice = ChatToolChoice::None;
    let default_prompt = processor
        .prepare_chat(&default_system_command)
        .expect("a request without a system message should use the template default")
        .rendered_prompt()
        .to_owned();
    assert!(default_prompt.contains(DEFAULT_SYSTEM_MESSAGE));

    let mut explicit_system_command = default_system_command.clone();
    explicit_system_command.request_id = astronomical_ipc_protocol::RequestId::new(9_811);
    explicit_system_command.messages.insert(
        0,
        ChatMessage::System {
            content: "Answer only with evidence from Romeo and Juliet.".to_owned(),
        },
    );
    let explicit_prompt = processor
        .prepare_chat(&explicit_system_command)
        .expect("an explicit system message should replace the default")
        .rendered_prompt()
        .to_owned();
    assert!(explicit_prompt.contains("Answer only with evidence from Romeo and Juliet."));
    assert!(!explicit_prompt.contains(DEFAULT_SYSTEM_MESSAGE));

    let mut empty_system_command = default_system_command;
    empty_system_command.request_id = astronomical_ipc_protocol::RequestId::new(9_812);
    empty_system_command.messages.insert(
        0,
        ChatMessage::System {
            content: String::new(),
        },
    );
    let empty_system_prompt = processor
        .prepare_chat(&empty_system_command)
        .expect("an empty system message should deliberately suppress the default")
        .rendered_prompt()
        .to_owned();
    assert!(!empty_system_prompt.contains(DEFAULT_SYSTEM_MESSAGE));
    assert!(!empty_system_prompt.contains("<system>"));
}

#[test]
fn should_preserve_prior_reasoning_only_when_the_template_contract_requests_it() {
    let prior_reasoning = "Romeo's choices drive the tragic conflict.";
    let prior_assistant = ChatMessage::Assistant {
        content: Some("The conflict is personal and social.".to_owned()),
        reasoning_content: Some(prior_reasoning.to_owned()),
        tool_calls: vec![ChatAssistantToolCall {
            id: "call-prior".to_owned(),
            function: ChatAssistantToolFunction {
                name: "find_character".to_owned(),
                arguments_json: r#"{"name":"Romeo"}"#.to_owned(),
            },
        }],
    };

    let mut preserving_command = romeo_and_juliet_command(9_813, Some(0));
    preserving_command
        .messages
        .insert(0, prior_assistant.clone());
    preserving_command.messages.insert(
        1,
        ChatMessage::Tool {
            tool_call_id: "call-prior".to_owned(),
            content: "Romeo appears throughout the supplied play.".to_owned(),
        },
    );
    let preserving_prompt = processor(SyntheticLagunaTextArtifact::small_included())
        .prepare_chat(&preserving_command)
        .expect("the preserving template should render prior reasoning")
        .rendered_prompt()
        .to_owned();
    assert!(preserving_prompt.contains(prior_reasoning));
    assert!(preserving_prompt.contains("<tool_call>find_character"));
    assert!(preserving_prompt.contains("<tool_response>"));
    assert!(preserving_prompt.contains("Romeo appears throughout the supplied play."));

    let mut omitting_command = romeo_and_juliet_command(9_814, Some(0));
    omitting_command.messages.insert(0, prior_assistant);
    let omitting_prompt = processor(SyntheticLagunaTextArtifact::extra_small_inline())
        .prepare_chat(&omitting_command)
        .expect("the non-preserving template should render the remaining history")
        .rendered_prompt()
        .to_owned();
    assert!(!omitting_prompt.contains(prior_reasoning));
}

#[test]
fn should_reject_image_input_for_the_text_only_laguna_contract() {
    let processor = processor(SyntheticLagunaTextArtifact::extra_small_inline());
    let mut chat_command = romeo_and_juliet_command(9_815, None);
    let ChatMessage::User { images, .. } = &mut chat_command.messages[0] else {
        panic!("the fixture must begin with a user message");
    };
    images.push(ChatImageInput {
        mime_type: "image/png".to_owned(),
        decoded_bytes: vec![137, 80, 78, 71],
    });

    let preparation_error = processor
        .prepare_chat(&chat_command)
        .expect_err("text-only Laguna must reject images before tokenization");

    assert!(matches!(
        preparation_error,
        LagunaPreparationError::ImageInputUnsupported
    ));
}

#[test]
fn should_enforce_validated_model_context_instead_of_the_tokenizer_sentinel() {
    let mut text_artifact = SyntheticLagunaTextArtifact::extra_small_inline();
    text_artifact.model_config["max_position_embeddings"] = json!(32);
    let processor = processor(text_artifact);

    let preparation_error = processor
        .prepare_chat(&romeo_and_juliet_command(9_816, None))
        .expect_err("the Romeo and Juliet prompt must exceed the 32-token model context");

    assert!(matches!(
        preparation_error,
        LagunaPreparationError::ContextLengthExceeded {
            maximum_context_tokens: 32,
            ..
        }
    ));
}

#[test]
fn should_validate_the_command_model_before_rendering_user_content() {
    let processor = processor(SyntheticLagunaTextArtifact::extra_small_inline());
    let mut chat_command = romeo_and_juliet_command(9_817, None);
    chat_command.model = "another-model".to_owned();

    let preparation_error = processor
        .prepare_chat(&chat_command)
        .expect_err("a command targeting another model must fail request-locally");

    assert!(matches!(
        preparation_error,
        LagunaPreparationError::ModelIdMismatch { .. }
    ));
}

#[test]
fn should_implement_the_architecture_neutral_generation_processor_contract() {
    // This compile-time boundary keeps Laguna usable by the neutral worker without another family.
    fn assert_generation_processor<Processor: ModelGenerationProcessor>() {}

    assert_generation_processor::<LagunaGenerationProcessor>();
}

fn processor(text_artifact: SyntheticLagunaTextArtifact) -> LagunaGenerationProcessor {
    LagunaGenerationProcessor::new(SYNTHETIC_LAGUNA_MODEL_ID, text_artifact.normalize())
        .expect("the validated synthetic Laguna descriptor should construct a processor")
}
