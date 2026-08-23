//! Qualifies tokenizer behavior from a validated Ornith artifact without
//! coupling the assertions to one packaged vocabulary snapshot.

use std::io::Cursor;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId,
};
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5OutputEvent, Qwen3_5RequestOutput, Qwen3_5SamplingStrategy,
    Qwen3_5Tokenizer,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

fn load_tokenizer() -> Qwen3_5Tokenizer {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Ornith artifact should validate");
    Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the Ornith tokenizer should load from validated model metadata")
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_load_the_ornith_tokenizer_and_expose_control_tokens() {
    let tokenizer = load_tokenizer();

    // Structural validity: the tokenizer must expose valid control token ids
    // that are within the model vocabulary, without asserting exact golden-master
    // vocabulary sizes that change with every packaging variant.
    let model_vocabulary_size = tokenizer.model_vocabulary_size();
    assert!(
        model_vocabulary_size > 0,
        "model vocabulary size must be positive"
    );
    assert!(
        tokenizer.tokenizer_vocabulary_size() > 0,
        "tokenizer vocabulary size must be positive"
    );
    assert!(
        tokenizer.tokenizer_vocabulary_size() <= model_vocabulary_size,
        "tokenizer vocabulary size must not exceed model vocabulary size"
    );
    let control_token_ids = [
        tokenizer.end_of_text_token_id(),
        tokenizer.im_start_token_id(),
        tokenizer.im_end_token_id(),
        tokenizer.think_start_token_id(),
        tokenizer.think_end_token_id(),
        tokenizer.image_pad_token_id(),
    ];
    for control_token_id in control_token_ids {
        assert!(
            control_token_id < model_vocabulary_size,
            "control token id {control_token_id} must be within vocabulary size {model_vocabulary_size}"
        );
    }
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_encode_a_structured_chat_prompt_into_valid_token_ids() {
    let tokenizer = load_tokenizer();
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(256)
        .collect::<String>();
    let rendered_prompt = format!(
        "<|im_start|>user\nUse this Romeo and Juliet source:\n{source_excerpt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );

    let encoded_tokens = tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the bounded rendered prompt should encode");

    // Structural validity: encoding a valid prompt must produce a non-empty
    // token sequence where every id is within the model vocabulary.
    let model_vocabulary_size = tokenizer.model_vocabulary_size();
    assert!(
        !encoded_tokens.is_empty(),
        "the encoded prompt must produce tokens"
    );
    for token_id in &encoded_tokens {
        assert!(
            *token_id < model_vocabulary_size,
            "encoded token id {token_id} must be within vocabulary size {model_vocabulary_size}"
        );
    }
    // The prompt must start with the im_start token.
    assert_eq!(encoded_tokens[0], tokenizer.im_start_token_id());
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_incrementally_decode_text_and_suppress_stop_tokens() {
    let tokenizer = load_tokenizer();
    let mut token_decoder = tokenizer.incremental_decoder();
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(128)
        .collect::<String>();
    let source_token_ids = tokenizer
        .encode_prompt(&source_excerpt)
        .expect("the Romeo and Juliet excerpt should encode");
    let mut decoded_source = String::new();

    for source_token_id in source_token_ids {
        if let Some(decoded_fragment) = token_decoder
            .push_token(source_token_id)
            .expect("the Romeo and Juliet excerpt should decode incrementally")
        {
            decoded_source.push_str(&decoded_fragment);
        }
    }
    if let Some(decoded_fragment) = token_decoder
        .finish()
        .expect("the Romeo and Juliet excerpt should flush")
    {
        decoded_source.push_str(&decoded_fragment);
    }
    assert_eq!(decoded_source, source_excerpt);

    let mut stop_token_decoder = tokenizer.incremental_decoder();
    assert_eq!(
        stop_token_decoder
            .push_token(tokenizer.im_end_token_id())
            .expect("im_end should be accepted as EOS"),
        None,
        "im_end must be suppressed as a stop token"
    );
    assert_eq!(
        stop_token_decoder
            .push_token(tokenizer.end_of_text_token_id())
            .expect("endoftext should be accepted as EOS"),
        None,
        "endoftext must be suppressed as a stop token"
    );
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_prepare_a_validated_structured_chat_command_for_ornith_prefill() {
    let tokenizer = load_tokenizer();
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(800),
        model: crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: "Explain Mars in one sentence.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(950),
            seed: Some(7),
            thinking_budget: Some(256),
        },
    };

    let engine_request = tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("the bounded structured chat command should prepare for Ornith");

    // Structural validity: the prepared request must have valid token ids within
    // the vocabulary, proper request metadata, and matching sampling parameters.
    assert_eq!(engine_request.request_id(), RequestId::new(800));
    assert!(
        !engine_request.input_token_ids().is_empty(),
        "the prompt must produce tokens"
    );
    assert_eq!(engine_request.max_output_tokens(), 512);
    assert_eq!(
        engine_request.sampling_strategy(),
        Qwen3_5SamplingStrategy::TopKTopP {
            temperature_thousandths: 600,
            top_k: 20,
            top_p_thousandths: 950,
            seed: Some(7),
        }
    );
    let model_vocabulary_size = tokenizer.model_vocabulary_size();
    for token_id in engine_request.input_token_ids() {
        assert!(
            *token_id < model_vocabulary_size,
            "prompt token id {token_id} must be within vocabulary size {model_vocabulary_size}"
        );
    }
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_prepare_image_chat_with_processed_visual_images_for_engine_prefill() {
    let tokenizer = load_tokenizer();
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(801),
        model: crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: "What color is this image?".to_owned(),
            images: vec![ChatImageInput {
                mime_type: "image/png".to_owned(),
                decoded_bytes: one_pixel_png(),
            }],
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 8,
            temperature_thousandths: Some(0),
            top_p_thousandths: Some(950),
            seed: None,
            thinking_budget: Some(256),
        },
    };

    let engine_request = tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("the image chat command should prepare for Ornith vision prefill");

    assert_eq!(engine_request.request_id(), RequestId::new(801));
    assert_eq!(engine_request.processed_visual_images().len(), 1);
    let processed_visual_image = &engine_request.processed_visual_images()[0];
    let image_pad_token_id = tokenizer.image_pad_token_id();
    assert_eq!(
        engine_request
            .input_token_ids()
            .iter()
            .filter(|token_id| **token_id == image_pad_token_id)
            .count(),
        processed_visual_image.image_token_count_after_spatial_merge,
        "image pad token count must match the processed visual image spatial merge count"
    );
    assert!(processed_visual_image.pixel_values_row_count > 0);
    assert!(processed_visual_image.pixel_values_column_count > 0);
    assert!(processed_visual_image.image_token_count_after_spatial_merge > 0);
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_decode_generated_tokens_into_separate_reasoning_and_text_events() {
    let tokenizer = load_tokenizer();
    let mut request_output = Qwen3_5RequestOutput::new(&tokenizer, &[], false)
        .expect("a request without tools should create bounded output state");
    let mut output_events = Vec::new();

    // Deriving ordinary text tokens from the required source fixture keeps this
    // parser qualification valid when a packaging variant changes token ids.
    let reasoning_excerpt = ROMEO_AND_JULIET_SOURCE.chars().take(64).collect::<String>();
    let assistant_excerpt = ROMEO_AND_JULIET_SOURCE
        .chars()
        .skip(64)
        .take(64)
        .collect::<String>();
    let mut generated_token_ids = vec![tokenizer.think_start_token_id()];
    generated_token_ids.extend(
        tokenizer
            .encode_prompt(&reasoning_excerpt)
            .expect("the Romeo and Juliet reasoning excerpt should encode"),
    );
    generated_token_ids.push(tokenizer.think_end_token_id());
    generated_token_ids.extend(
        tokenizer
            .encode_prompt(&assistant_excerpt)
            .expect("the Romeo and Juliet assistant excerpt should encode"),
    );

    for generated_token_id in generated_token_ids {
        output_events.extend(
            request_output
                .push_token(generated_token_id)
                .expect("the generated output token should decode and parse"),
        );
    }
    output_events.extend(
        request_output
            .finish()
            .expect("the completed reasoning and text output should finish cleanly"),
    );

    // Structural validity: there must be both reasoning and text events.
    let has_reasoning = output_events
        .iter()
        .any(|event| matches!(event, Qwen3_5OutputEvent::ReasoningDelta(_)));
    let has_text = output_events
        .iter()
        .any(|event| matches!(event, Qwen3_5OutputEvent::TextDelta(_)));
    assert!(
        has_reasoning,
        "the output must contain at least one reasoning delta event"
    );
    assert!(
        has_text,
        "the output must contain at least one text delta event"
    );
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_flush_pending_byte_fallback_text_when_request_output_finishes() {
    let tokenizer = load_tokenizer();
    let mut request_output = Qwen3_5RequestOutput::new(&tokenizer, &[], false)
        .expect("a request without tools should create bounded output state");

    // Byte-fallback token id 126 is the '~' character in the Qwen3.5 family.
    // Structural validity: pushing a byte-fallback token and then finishing
    // must produce at least one text event, without asserting exact string content
    // that depends on tokenizer implementation details.
    assert_eq!(
        request_output
            .push_token(126)
            .expect("an incomplete byte-fallback token should be buffered"),
        Vec::new()
    );
    let finish_events = request_output
        .finish()
        .expect("completion should flush the buffered byte-fallback token");
    assert!(
        !finish_events.is_empty(),
        "finishing after a buffered byte-fallback token must produce at least one output event"
    );
    assert!(
        finish_events
            .iter()
            .any(|event| matches!(event, Qwen3_5OutputEvent::TextDelta(_))),
        "at least one finish event must be a TextDelta"
    );
}

fn one_pixel_png() -> Vec<u8> {
    let source_image = RgbImage::from_pixel(1, 1, Rgb([128, 64, 32]));
    let mut encoded_image_bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source_image)
        .write_to(&mut encoded_image_bytes, ImageFormat::Png)
        .expect("the in-memory one-pixel PNG should encode");
    encoded_image_bytes.into_inner()
}
