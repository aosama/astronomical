//! The folded ternary family's acceptance journeys over the shipped pack.
//!
//! These are acceptance, not unit, contracts: every test loads the real
//! installed pack (roughly 8 GB of model weights) and drives the
//! graphics processor, so they are feature-gated on `direct-mlx`, run
//! `--ignored` by default, and locate the pack through environment variables
//! instead of any hardcoded developer path. The performance tiers live in
//! `prism_hadamard/` beside the shared pack and prompt helpers.

use serial_test::serial;

#[path = "../support/romeo_and_juliet.rs"]
mod romeo_and_juliet;

#[path = "prism_hadamard/mod.rs"]
mod prism_hadamard;

use prism_hadamard::support::{loaded_journey_model, pack_directory, parity_reference_path};

/// How many greedy tokens the parity journey replays (the reference dump's
/// length).
const PARITY_TOKEN_COUNT: usize = 32;

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and replays the upstream greedy trajectory on the graphics processor"]
fn should_replay_the_upstream_reference_greedy_trajectory() {
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(parity_reference_path())
            .expect("the parity reference file should be readable"),
    )
    .expect("the parity reference file should be valid JSON");
    let reference_prompt = reference["prompt"]
        .as_str()
        .expect("the reference should carry its source prompt");
    let expected_prompt = romeo_and_juliet::summary_prompt(120);
    assert_eq!(reference_prompt, expected_prompt);
    let tokenizer = tokenizers::Tokenizer::from_file(pack_directory().join("tokenizer.json"))
        .expect("the pack tokenizer should load");
    let rendered_prompt = format!(
        "<|im_start|>user\n{reference_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let expected_input_ids = tokenizer
        .encode(rendered_prompt.as_str(), false)
        .expect("the reference source prompt should encode")
        .get_ids()
        .iter()
        .map(|token_id| *token_id as i32)
        .collect::<Vec<_>>();
    let input_ids: Vec<i32> = reference["input_ids"]
        .as_array()
        .expect("the reference should carry input_ids")
        .iter()
        .map(|token| token.as_i64().expect("token id") as i32)
        .collect();
    assert_eq!(input_ids, expected_input_ids);
    let expected_tokens: Vec<u32> = reference["greedy_token_ids"]
        .as_array()
        .expect("the reference should carry greedy_token_ids")
        .iter()
        .map(|token| token.as_i64().expect("token id") as u32)
        .collect();
    assert_eq!(expected_tokens.len(), PARITY_TOKEN_COUNT);

    let model = loaded_journey_model();
    // An empty stop set makes the production generation loop decode exactly
    // the reference length; any arithmetic divergence anywhere in the
    // trajectory breaks the sequence equality below.
    let actual_tokens = model
        .generate_greedy(&input_ids, PARITY_TOKEN_COUNT, &[])
        .expect("the parity replay should generate");
    assert_eq!(
        actual_tokens, expected_tokens,
        "the family model must reproduce the upstream greedy trajectory token for token"
    );
}

#[test]
#[serial]
#[ignore = "loads the real roughly 8 GB folded pack and generates a visible reply on the graphics processor"]
fn should_receive_text_and_reply_with_text() {
    let tokenizer = tokenizers::Tokenizer::from_file(pack_directory().join("tokenizer.json"))
        .expect("the pack tokenizer should load");
    let end_of_turn_id = tokenizer
        .token_to_id("<|im_end|>")
        .expect("the pack tokenizer should declare <|im_end|>") as u32;
    let end_of_text_id = tokenizer
        .token_to_id("<|endoftext|>")
        .expect("the pack tokenizer should declare <|endoftext|>") as u32;

    let user_prompt = romeo_and_juliet::summary_prompt(120);
    // The single-user-turn, thinking-disabled render of the pack's chat
    // template, spelled out because this direct text journey renders no jinja.
    let rendered_prompt = format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let prompt_token_ids: Vec<i32> = tokenizer
        .encode(rendered_prompt.as_str(), false)
        .expect("the rendered prompt should encode")
        .get_ids()
        .iter()
        .map(|token_id| *token_id as i32)
        .collect();

    let model = loaded_journey_model();
    let generated_token_ids = model
        .generate_greedy(&prompt_token_ids, 64, &[end_of_turn_id, end_of_text_id])
        .expect("the text-generation journey should run");
    let reply_text = tokenizer
        .decode(&generated_token_ids, false)
        .expect("the generated ids should decode");

    println!("Romeo-and-Juliet reply: {reply_text:?}");
    assert!(
        !generated_token_ids.is_empty(),
        "the model must produce at least one visible token"
    );
    assert!(
        !reply_text.trim().is_empty(),
        "the decoded reply must carry visible text"
    );
}
