use std::io::Cursor;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatImageInput, ChatMessage, ChatToolChoice,
    RequestId,
};
use astronomical_model_serving::{
    Qwen3_5MoEArtifactValidator, Qwen3_5MoEOutputEvent, Qwen3_5MoERequestOutput,
    Qwen3_5MoESamplingStrategy, Qwen3_5MoETokenizer,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

const IMAGE_PAD_TOKEN_ID: u32 = 248_056;

fn load_tokenizer() -> Qwen3_5MoETokenizer {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5MoEArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate");
    Qwen3_5MoETokenizer::from_validated_artifact(&validated_artifact)
        .expect("the pinned Ornith tokenizer should load from validated model metadata")
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_load_the_pinned_ornith_tokenizer_and_certified_control_tokens() {
    let tokenizer = load_tokenizer();

    assert_eq!(tokenizer.tokenizer_vocabulary_size(), 248_077);
    assert_eq!(tokenizer.model_vocabulary_size(), 248_320);
    assert_eq!(tokenizer.end_of_text_token_id(), 248_044);
    assert_eq!(tokenizer.im_start_token_id(), 248_045);
    assert_eq!(tokenizer.im_end_token_id(), 248_046);
    assert_eq!(tokenizer.think_start_token_id(), 248_068);
    assert_eq!(tokenizer.think_end_token_id(), 248_069);
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_match_the_independent_ornith_prompt_token_vector() {
    let tokenizer = load_tokenizer();
    let rendered_prompt = concat!(
        "<|im_start|>user\n",
        "Explain Mars in one sentence.",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "<think>\n",
    );

    assert_eq!(
        tokenizer
            .encode_prompt(rendered_prompt)
            .expect("the bounded rendered prompt should encode"),
        vec![
            248_045, 846, 198, 814, 20_139, 20_403, 303, 799, 11_316, 13, 248_046, 198, 248_045,
            74_455, 198, 248_068, 198,
        ]
    );
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_incrementally_decode_only_new_ornith_text_suffixes_and_suppress_stop_tokens() {
    let tokenizer = load_tokenizer();
    let mut token_decoder = tokenizer.incremental_decoder();

    assert_eq!(
        token_decoder
            .push_token(6_918)
            .expect("alpha should decode"),
        Some("alpha".to_owned())
    );
    assert_eq!(
        token_decoder
            .push_token(13_053)
            .expect("beta should decode"),
        Some(" beta".to_owned())
    );
    assert_eq!(
        token_decoder
            .push_token(248_046)
            .expect("im_end should be accepted as EOS"),
        None
    );
    assert_eq!(
        token_decoder
            .push_token(248_044)
            .expect("endoftext should be accepted as EOS"),
        None
    );
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_prepare_a_validated_structured_chat_command_for_ornith_prefill() {
    let tokenizer = load_tokenizer();
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(800),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
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
            thinking_budget: None,
        },
    };

    let engine_request = tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("the bounded structured chat command should prepare for Ornith");

    assert_eq!(engine_request.request_id(), RequestId::new(800));
    assert_eq!(engine_request.input_token_ids().len(), 17);
    assert_eq!(engine_request.max_output_tokens(), 512);
    assert_eq!(
        engine_request.sampling_strategy(),
        Qwen3_5MoESamplingStrategy::TopKTopP {
            temperature_thousandths: 600,
            top_k: 20,
            top_p_thousandths: 950,
            seed: Some(7),
        }
    );
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_prepare_image_chat_with_processed_visual_images_for_engine_prefill() {
    let tokenizer = load_tokenizer();
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(801),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
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
            thinking_budget: None,
        },
    };

    let engine_request = tokenizer
        .prepare_chat(&chat_generation_command, true)
        .expect("the image chat command should prepare for Ornith vision prefill");

    assert_eq!(engine_request.request_id(), RequestId::new(801));
    assert_eq!(
        engine_request
            .input_token_ids()
            .iter()
            .filter(|token_id| **token_id == IMAGE_PAD_TOKEN_ID)
            .count(),
        64
    );
    assert_eq!(engine_request.processed_visual_images().len(), 1);
    let processed_visual_image = &engine_request.processed_visual_images()[0];
    assert_eq!(processed_visual_image.pixel_values_row_count, 256);
    assert_eq!(processed_visual_image.pixel_values_column_count, 1_536);
    assert_eq!(
        processed_visual_image.image_token_count_after_spatial_merge,
        64
    );
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_decode_generated_tokens_into_separate_reasoning_and_text_events() {
    let tokenizer = load_tokenizer();
    let mut request_output = Qwen3_5MoERequestOutput::new(&tokenizer, &[], false)
        .expect("a request without tools should create bounded output state");
    let mut output_events = Vec::new();

    for generated_token_id in [248_068, 198, 1_960, 198, 248_069, 271, 16_936, 13] {
        output_events.extend(
            request_output
                .push_token(generated_token_id)
                .expect("the certified output token should decode and parse"),
        );
    }
    output_events.extend(
        request_output
            .finish()
            .expect("the completed reasoning and text output should finish cleanly"),
    );

    let reasoning_text = output_events
        .iter()
        .filter_map(|output_event| match output_event {
            Qwen3_5MoEOutputEvent::ReasoningDelta(reasoning_delta) => {
                Some(reasoning_delta.as_str())
            }
            Qwen3_5MoEOutputEvent::TextDelta(_)
            | Qwen3_5MoEOutputEvent::ToolCall(_)
            | Qwen3_5MoEOutputEvent::ModelVisibleCorrection { .. } => None,
        })
        .collect::<String>();
    let assistant_text = output_events
        .iter()
        .filter_map(|output_event| match output_event {
            Qwen3_5MoEOutputEvent::TextDelta(text_delta) => Some(text_delta.as_str()),
            Qwen3_5MoEOutputEvent::ReasoningDelta(_)
            | Qwen3_5MoEOutputEvent::ToolCall(_)
            | Qwen3_5MoEOutputEvent::ModelVisibleCorrection { .. } => None,
        })
        .collect::<String>();
    assert_eq!(reasoning_text, "\ncheck\n");
    assert_eq!(assistant_text, "\n\nDone.");
}

#[test]
#[ignore = "requires model_directories to discover Ornith-1.0-35B-OptiQ-4bit"]
fn should_flush_pending_byte_fallback_text_when_request_output_finishes() {
    let tokenizer = load_tokenizer();
    let mut request_output = Qwen3_5MoERequestOutput::new(&tokenizer, &[], false)
        .expect("a request without tools should create bounded output state");

    assert_eq!(
        request_output
            .push_token(126)
            .expect("an incomplete byte-fallback token should be buffered"),
        Vec::new()
    );
    assert_eq!(
        request_output
            .finish()
            .expect("completion should flush the buffered byte-fallback token"),
        vec![Qwen3_5MoEOutputEvent::TextDelta("�".to_owned())]
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
