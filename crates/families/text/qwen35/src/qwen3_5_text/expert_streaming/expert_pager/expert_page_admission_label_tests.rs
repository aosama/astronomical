//! The name a streamed page carries in its admission decision: a refusal must
//! say how many experts the page held, how many tokens routed it and which
//! phase asked, so a whole-layer read is never mistaken for one token's page.

use super::expert_page_streaming::streamed_page_admission_label;
use model_primitives::serving_phase::SessionStreamingPhase;

#[test]
fn should_name_the_layer_the_expert_count_the_routed_tokens_and_the_phase() {
    assert_eq!(
        streamed_page_admission_label(9, 237, 4, SessionStreamingPhase::Decode),
        "rust_streamed_expert_layer_9 (237 experts, 4 routed tokens, Decode)"
    );
}

#[test]
fn should_name_a_prompt_chunk_page_as_prefill() {
    assert_eq!(
        streamed_page_admission_label(0, 256, 4096, SessionStreamingPhase::Prefill),
        "rust_streamed_expert_layer_0 (256 experts, 4096 routed tokens, Prefill)"
    );
}

#[test]
fn should_name_an_empty_page_without_failing() {
    assert_eq!(
        streamed_page_admission_label(39, 0, 1, SessionStreamingPhase::Decode),
        "rust_streamed_expert_layer_39 (0 experts, 1 routed tokens, Decode)"
    );
}
