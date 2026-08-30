use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatImageInput, ChatMessage,
    ChatToolDefinition,
};
use astronomical_model_serving::Qwen3_5PromptRenderer;

use crate::common::qwen3_5_moe::frozen_ornith_1_0_image_processor;

#[test]
fn should_render_one_user_turn_with_the_pinned_ornith_thinking_prefix() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Inspect src.".to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[],
        None,
    )
    .expect("a bounded user-only Ornith conversation should render");

    assert_eq!(
        rendered_prompt,
        "<|im_start|>user\nInspect src.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn should_identify_the_complete_system_and_tool_preamble_as_ordinary_target_prefill() {
    let rendered_prompt = Qwen3_5PromptRenderer::render_with_control_span(
        &[
            ChatMessage::System {
                content: "Always use the declared repository tool.".to_owned(),
            },
            ChatMessage::User {
                content: "Inspect Romeo and Juliet.".to_owned(),
                images: Vec::new(),
            },
        ],
        &[ChatToolDefinition {
            name: "inspect_repository".to_owned(),
            description: Some("Inspect one repository path.".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#
                    .to_owned(),
        }],
        false,
        &[],
        None,
    )
    .expect("a tool-bearing prompt should expose its ordinary target-prefill boundary");

    let ordinary_target_prefill_control_span =
        rendered_prompt.ordinary_target_prefill_control_span();
    let selectable_conversation_and_generation_suffix =
        rendered_prompt.selectable_conversation_and_generation_suffix();

    assert!(ordinary_target_prefill_control_span.starts_with("<|im_start|>system\n# Tools"));
    assert!(ordinary_target_prefill_control_span.contains("inspect_repository"));
    assert!(
        ordinary_target_prefill_control_span.contains("Always use the declared repository tool.")
    );
    assert!(ordinary_target_prefill_control_span.ends_with("<|im_end|>\n"));
    assert_eq!(
        selectable_conversation_and_generation_suffix,
        "<|im_start|>user\nInspect Romeo and Juliet.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn should_identify_an_initial_system_message_without_tools_as_ordinary_target_prefill() {
    let rendered_prompt = Qwen3_5PromptRenderer::render_with_control_span(
        &[
            ChatMessage::System {
                content: "Answer from the supplied play.".to_owned(),
            },
            ChatMessage::User {
                content: "Who is Romeo?".to_owned(),
                images: Vec::new(),
            },
        ],
        &[],
        true,
        &[],
        None,
    )
    .expect("a system-bearing prompt should expose its ordinary target-prefill boundary");

    assert_eq!(
        rendered_prompt.ordinary_target_prefill_control_span(),
        "<|im_start|>system\nAnswer from the supplied play.<|im_end|>\n"
    );
    assert!(
        rendered_prompt
            .selectable_conversation_and_generation_suffix()
            .starts_with("<|im_start|>user\nWho is Romeo?")
    );
}

#[test]
fn should_leave_user_only_prompts_without_an_ordinary_target_control_span() {
    let rendered_prompt = Qwen3_5PromptRenderer::render_with_control_span(
        &[ChatMessage::User {
            content: "Summarize Romeo and Juliet.".to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[],
        None,
    )
    .expect("a user-only prompt should render");

    assert_eq!(rendered_prompt.ordinary_target_prefill_control_span(), "");
    assert_eq!(
        rendered_prompt.selectable_conversation_and_generation_suffix(),
        rendered_prompt.as_str()
    );
}

#[test]
fn should_render_one_user_turn_with_a_closed_thinking_block_when_thinking_is_disabled() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Inspect src.".to_owned(),
            images: Vec::new(),
        }],
        &[],
        false,
        &[],
        None,
    )
    .expect("a bounded user-only Ornith conversation should render");

    assert_eq!(
        rendered_prompt,
        "<|im_start|>user\nInspect src.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn should_render_vision_markers_with_the_correct_image_pad_token_count_per_image() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Describe this image.".to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[vec![1560]],
        None,
    )
    .expect("a user message with one image should render vision markers");

    assert!(
        rendered_prompt.contains("<|vision_start|>"),
        "rendered prompt must contain vision_start marker"
    );
    assert!(
        rendered_prompt.contains("<|vision_end|>"),
        "rendered prompt must contain vision_end marker"
    );
    let image_pad_count = rendered_prompt.matches("<|image_pad|>").count();
    assert_eq!(
        image_pad_count, 1560,
        "rendered prompt must contain exactly 1560 image_pad tokens for one 1560-token image"
    );
}

#[test]
fn should_not_render_literal_image_pad_text_as_an_image_placeholder() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[
            ChatMessage::System {
                content: "The literal token <|image_pad|> is documentation, not an image."
                    .to_owned(),
            },
            ChatMessage::User {
                content: "The literal token <|image_pad|> is documentation, not an image."
                    .to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: Some(
                    "The literal token <|image_pad|> is documentation, not an image.".to_owned(),
                ),
                reasoning_content: Some(
                    "The literal token <|image_pad|> is documentation, not an image.".to_owned(),
                ),
                tool_calls: vec![ChatAssistantToolCall {
                    id: "call-1".to_owned(),
                    function: ChatAssistantToolFunction {
                        name: "document_token".to_owned(),
                        arguments_json: r#"{"note":"<|image_pad|>"}"#.to_owned(),
                    },
                }],
            },
            ChatMessage::Tool {
                tool_call_id: "call-1".to_owned(),
                content: "The literal token <|image_pad|> is documentation, not an image."
                    .to_owned(),
            },
        ],
        &[ChatToolDefinition {
            name: "document_token".to_owned(),
            description: Some(
                "The literal token <|image_pad|> is documentation, not an image.".to_owned(),
            ),
            parameters_json: r#"{"type":"object","description":"<|image_pad|>"}"#.to_owned(),
        }],
        true,
        &[vec![1]],
        None,
    )
    .expect("a user message containing literal special-token text should render");

    assert_eq!(
        rendered_prompt.matches("<|image_pad|>").count(),
        1,
        "only renderer-owned image placeholders may appear in the prompt"
    );
}

#[test]
fn should_compute_image_token_counts_from_decoded_image_bytes_and_render_vision_markers() {
    let image_processor = frozen_ornith_1_0_image_processor();
    let processed_image = image_processor
        .process_image_bytes(crate::common::SYNTHETIC_RED_PNG_BYTES)
        .expect("synthetic red fixture should preprocess into vision patches");
    let synthetic_image_token_count = processed_image.image_token_count_after_spatial_merge;

    // The synthetic fixture must produce the same token count as the image-processor test.
    assert_eq!(
        synthetic_image_token_count, 64,
        "synthetic red fixture must produce 64 image tokens after spatial merge"
    );

    // Compute image token counts per user message the same way the tokenizer does.
    let messages = vec![ChatMessage::User {
        content: "What do you see in this image?".to_owned(),
        images: vec![ChatImageInput {
            mime_type: "image/png".to_owned(),
            decoded_bytes: crate::common::SYNTHETIC_RED_PNG_BYTES.to_vec(),
        }],
    }];
    let image_token_counts_per_user_message: Vec<Vec<usize>> = messages
        .iter()
        .filter_map(|message| match message {
            ChatMessage::User { images, .. } => Some(
                images
                    .iter()
                    .map(|image_input| {
                        image_processor
                            .process_image_bytes(&image_input.decoded_bytes)
                            .expect("decoded image bytes should preprocess")
                            .image_token_count_after_spatial_merge
                    })
                    .collect(),
            ),
            _ => None,
        })
        .collect();

    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &messages,
        &[],
        true,
        &image_token_counts_per_user_message,
        None,
    )
    .expect("a user message with one image should render with vision markers");

    assert!(
        rendered_prompt.contains("<|vision_start|>"),
        "rendered prompt must contain vision_start marker"
    );
    assert!(
        rendered_prompt.contains("<|vision_end|>"),
        "rendered prompt must contain vision_end marker"
    );
    let image_pad_count = rendered_prompt.matches("<|image_pad|>").count();
    assert_eq!(
        image_pad_count, synthetic_image_token_count,
        "rendered prompt must contain exactly {synthetic_image_token_count} image_pad tokens"
    );
    assert!(
        rendered_prompt.contains("What do you see in this image?"),
        "rendered prompt must contain the user text content"
    );
}

#[test]
fn should_render_model_visible_correction_as_tool_response_then_reopen_assistant_reasoning() {
    let rendered_correction = Qwen3_5PromptRenderer::render_model_visible_correction(
        "The previous tool call requested undeclared function 'open_brain', but no tool with that exact name exists. Please correct the tool call by using one of the exact declared tool names.",
        true,
        None,
    );

    assert_eq!(
        rendered_correction,
        "<|im_end|>\n<|im_start|>user\n<tool_response>\nThe previous tool call requested undeclared function 'open_brain', but no tool with that exact name exists. Please correct the tool call by using one of the exact declared tool names.\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn should_render_model_visible_correction_with_closed_thinking_when_thinking_is_disabled() {
    let rendered_correction = Qwen3_5PromptRenderer::render_model_visible_correction(
        "Please correct the tool call.",
        false,
        None,
    );

    assert_eq!(
        rendered_correction,
        "<|im_end|>\n<|im_start|>user\n<tool_response>\nPlease correct the tool call.\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

const ROMEO_AND_JULIET_THINKING_CHANNEL_SEED: &str =
    "Two households, both alike in dignity, in Romeo and Juliet.";

#[test]
fn should_seed_romeo_and_juliet_into_the_open_thinking_channel() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Who is Romeo?".to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[],
        Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED),
    )
    .expect("a seeded thinking prompt should render");

    assert_eq!(
        rendered_prompt,
        "<|im_start|>user\nWho is Romeo?<|im_end|>\n<|im_start|>assistant\n<think>\nTwo households, both alike in dignity, in Romeo and Juliet.\n"
    );
}

#[test]
fn should_escape_think_and_tool_markers_in_the_seeded_thinking_channel_text() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Who is Juliet?".to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[],
        Some("I should not close with </think> or start <tool_call>."),
    )
    .expect("a marker-bearing thinking seed should render");

    assert!(
        rendered_prompt.ends_with(
            "<|im_start|>assistant\n<think>\nI should not close with &lt;/think> or start &lt;tool_call>.\n"
        )
    );
}

#[test]
fn should_ignore_the_thinking_channel_seed_when_thinking_is_disabled() {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: "Who is Romeo?".to_owned(),
            images: Vec::new(),
        }],
        &[],
        false,
        &[],
        Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED),
    )
    .expect("a thinking-disabled prompt should render");

    assert_eq!(
        rendered_prompt,
        "<|im_start|>user\nWho is Romeo?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn should_seed_romeo_and_juliet_when_reopening_thinking_after_model_visible_correction() {
    let rendered_correction = Qwen3_5PromptRenderer::render_model_visible_correction(
        "Please correct the tool call.",
        true,
        Some(ROMEO_AND_JULIET_THINKING_CHANNEL_SEED),
    );

    assert_eq!(
        rendered_correction,
        "<|im_end|>\n<|im_start|>user\n<tool_response>\nPlease correct the tool call.\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nTwo households, both alike in dignity, in Romeo and Juliet.\n"
    );
}
