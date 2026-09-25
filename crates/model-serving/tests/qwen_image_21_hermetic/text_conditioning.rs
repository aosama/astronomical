//! Hermetic tests for the Qwen-Image-2.1 text-conditioning packaging pipeline.
//!
//! The oracle fixture (`tests/fixtures/qwen_image_21/conditioning_oracle.json`) records
//! the raw tokenized template ids + attention masks captured from the real
//! `mlx-community/Qwen-Image-2.1-MLX-4bit` processor tokenizer, alongside the reference
//! pipeline's packaged outputs. These tests feed the raw ids/mask through
//! [`build_prompt_conditioning`] and require the packaged `(B, L)` outputs to match the
//! reference element-for-element. This pins the highest "do not invent" risk of the
//! conditioning path: the drop-index heuristic, the padding split, the image-pad mask,
//! and the post-drop right-padding, all bit-exact against the diffusers oracle.
//!
//! The end-to-end counterpart (real tokenizer consuming the Rust-rendered templates)
//! lives in `installed_model_contracts/qwen_image_21_conditioning.rs`.
//!
//! The template-rendering and validation tests below complete the same contract without the
//! fixture: they pin the exact template shape (byte-level), the special-token bracket bytes, the
//! empty-prompt rule, and the typed rejection paths. They build expectations from the public
//! `special_tokens` constants rather than from the rendering code itself, so they cannot pass
//! tautologically.

use astronomical_model_serving::{
    PromptConditioningOutput, QWEN_IMAGE_21_SYS_PROMPT, TextConditioningError,
    build_prompt_conditioning, normalize_empty_prompt, render_t2i_prompt_template,
    render_ti2i_prompt_template, render_ti2i_prompt_template_with_image_count, special_tokens,
};
use serde_json::Value;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

const ORACLE_FIXTURE_JSON: &str =
    include_str!("../fixtures/qwen_image_21/conditioning_oracle.json");

struct OracleCase {
    raw_input_ids: Vec<Vec<u32>>,
    raw_attention_mask: Vec<Vec<u32>>,
    expected_condition_ids: Vec<Vec<u32>>,
    expected_attention_mask: Vec<Vec<u8>>,
    expected_image_pad_mask: Vec<Vec<u8>>,
}

struct OracleFixture {
    drop_idx: usize,
    image_pad_token_id: u32,
    pad_token_id: u32,
    t2i: OracleCase,
    ti2i: OracleCase,
}

fn u32_rows(value: &Value, field: &str) -> Vec<Vec<u32>> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("oracle fixture {field} must be an array"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("oracle fixture {field} rows must be arrays"))
                .iter()
                .map(|element| {
                    element.as_u64().unwrap_or_else(|| {
                        panic!("oracle fixture {field} must hold unsigned integers")
                    }) as u32
                })
                .collect()
        })
        .collect()
}

fn u8_rows(value: &Value, field: &str) -> Vec<Vec<u8>> {
    u32_rows(value, field)
        .into_iter()
        .map(|row| row.into_iter().map(|element| element as u8).collect())
        .collect()
}

fn oracle_case(value: &Value, name: &str) -> OracleCase {
    OracleCase {
        raw_input_ids: u32_rows(value, "raw_input_ids"),
        raw_attention_mask: u32_rows(value, "raw_attention_mask"),
        expected_condition_ids: u32_rows(value, "expected_condition_ids"),
        expected_attention_mask: u8_rows(value, "expected_attention_mask"),
        expected_image_pad_mask: u8_rows(value, "expected_image_pad_mask"),
    }
    .with_name_checked(name)
}

impl OracleCase {
    fn with_name_checked(self, name: &str) -> Self {
        assert_eq!(
            self.raw_input_ids.len(),
            self.raw_attention_mask.len(),
            "{name}: oracle raw ids and masks must have equal batch length"
        );
        assert_eq!(
            self.expected_condition_ids.len(),
            self.expected_attention_mask.len(),
            "{name}: oracle expected ids and masks must have equal batch length"
        );
        assert_eq!(
            self.expected_condition_ids.len(),
            self.expected_image_pad_mask.len(),
            "{name}: oracle expected ids and image-pad masks must have equal batch length"
        );
        self
    }
}

fn oracle_fixture() -> &'static OracleFixture {
    static FIXTURE: OnceLock<OracleFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let value: Value = serde_json::from_str(ORACLE_FIXTURE_JSON)
            .expect("the oracle conditioning fixture must parse as JSON");
        OracleFixture {
            drop_idx: value["drop_idx"]
                .as_u64()
                .expect("drop_idx must be unsigned") as usize,
            image_pad_token_id: value["image_pad_token_id"]
                .as_u64()
                .expect("image_pad_token_id must be unsigned")
                as u32,
            pad_token_id: value["pad_token_id"]
                .as_u64()
                .expect("pad_token_id must be unsigned") as u32,
            t2i: oracle_case(&value["t2i"], "t2i"),
            ti2i: oracle_case(&value["ti2i"], "ti2i"),
        }
    })
}

fn assert_matches_reference(output: PromptConditioningOutput, expected: &OracleCase, case: &str) {
    assert_eq!(
        output.condition_ids, expected.expected_condition_ids,
        "{case}: condition ids must match the oracle exactly"
    );
    assert_eq!(
        output.encoder_attention_mask, expected.expected_attention_mask,
        "{case}: encoder attention mask must match the oracle exactly"
    );
    assert_eq!(
        output.image_pad_mask, expected.expected_image_pad_mask,
        "{case}: image-pad mask must match the oracle exactly"
    );
}

#[test]
fn should_match_the_oracle_t2i_packaging_exactly() {
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.t2i.raw_input_ids,
        &fixture.t2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle T2I batch must package successfully");
    assert_matches_reference(output, &fixture.t2i, "t2i");
}

#[test]
fn should_match_the_oracle_ti2i_packaging_exactly() {
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.ti2i.raw_input_ids,
        &fixture.ti2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle ti2i batch must package successfully");
    assert_matches_reference(output, &fixture.ti2i, "ti2i");
}

#[test]
fn should_pack_the_t2i_batch_to_the_post_drop_maximum_length() {
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.t2i.raw_input_ids,
        &fixture.t2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle T2I batch must package successfully");
    assert_eq!(
        output.seq_len(),
        fixture.t2i.expected_condition_ids[0].len()
    );
}

#[test]
fn should_pack_the_ti2i_batch_to_the_post_drop_maximum_length() {
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.ti2i.raw_input_ids,
        &fixture.ti2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle ti2i batch must package successfully");
    assert_eq!(
        output.seq_len(),
        fixture.ti2i.expected_condition_ids[0].len()
    );
}

#[test]
fn should_drop_exactly_the_leading_system_block_from_the_t2i_batch() {
    // The first `drop_idx` valid tokens of every T2I sample are the system block
    // (im_start + system + newline + sys prompt + im_end + newline). After the drop the
    // first kept token must be the user-turn `<|im_start|>` marker (151644).
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.t2i.raw_input_ids,
        &fixture.t2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle T2I batch must package successfully");

    for (sample_index, sample) in output.condition_ids.iter().enumerate() {
        let first_valid = sample
            .iter()
            .zip(output.encoder_attention_mask[sample_index].iter())
            .find(|(_, mask)| **mask == 1)
            .map(|(&id, _)| id)
            .expect("every T2I sample keeps at least one valid token");
        assert_eq!(
            first_valid, 151_644,
            "sample {sample_index} must start at the user-turn im_start after the system drop"
        );
    }
}

#[test]
fn should_mark_exactly_one_image_position_per_ti2i_sample() {
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.ti2i.raw_input_ids,
        &fixture.ti2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle ti2i batch must package successfully");

    for (sample_index, image_mask) in output.image_pad_mask.iter().enumerate() {
        let marked: Vec<usize> = image_mask
            .iter()
            .enumerate()
            .filter(|(_, mask)| **mask == 1)
            .map(|(position, _)| position)
            .collect();
        assert_eq!(
            marked.len(),
            1,
            "ti2i sample {sample_index} carries exactly one text-only reference-image slot"
        );
        assert_eq!(
            output.condition_ids[sample_index][marked[0]], fixture.image_pad_token_id,
            "the marked position must hold the image-pad token id"
        );
    }
}

#[test]
fn should_only_pad_raw_rows_with_the_declared_pad_token() {
    // Left-padding positions (mask 0) must carry the tokenizer's pad id, so the e2e
    // counterpart can reproduce the processor's padding side exactly.
    let fixture = oracle_fixture();
    for (case_name, case) in [("t2i", &fixture.t2i), ("ti2i", &fixture.ti2i)] {
        for (sample_index, (ids, masks)) in case
            .raw_input_ids
            .iter()
            .zip(case.raw_attention_mask.iter())
            .enumerate()
        {
            for (position, (&id, &mask)) in ids.iter().zip(masks.iter()).enumerate() {
                if mask == 0 {
                    assert_eq!(
                        id, fixture.pad_token_id,
                        "{case_name} sample {sample_index} position {position}: masked-out ids must be the pad token"
                    );
                }
            }
        }
    }
}

#[test]
fn should_keep_every_kept_t2i_token_valid_in_the_encoder_mask() {
    // The reference builds the encoder attention mask from torch.ones over the kept
    // tokens, so every pre-padding position must be 1 even where the raw tokenizer mask
    // had left-padding zeros.
    let fixture = oracle_fixture();
    let output = build_prompt_conditioning(
        &fixture.t2i.raw_input_ids,
        &fixture.t2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle T2I batch must package successfully");

    for (sample_index, mask) in output.encoder_attention_mask.iter().enumerate() {
        let raw_valid_count = fixture.t2i.raw_attention_mask[sample_index]
            .iter()
            .filter(|&value| *value > 0)
            .count();
        let kept_length = raw_valid_count - fixture.drop_idx;
        assert!(
            mask.iter().take(kept_length).all(|&value| value == 1),
            "sample {sample_index} mask must be ones over its kept prefix"
        );
        assert!(
            mask.iter().skip(kept_length).all(|&value| value == 0),
            "sample {sample_index} mask must be zeros over its right padding"
        );
    }
}

// ---- Template rendering (no fixture needed) ----------------------------------------------

/// The `<imageN><|vision_start|><|image_pad|><|vision_end|>` block the reference inserts per
/// reference image, composed from the public special tokens so the expectation is independent
/// of the rendering code under test.
fn expected_image_block(image_number: usize) -> String {
    format!(
        "\u{3c}image{image_number}\u{3e}{}{}{}",
        special_tokens::VISION_START,
        special_tokens::IMAGE_PAD,
        special_tokens::VISION_END
    )
}

#[test]
fn should_compile_special_tokens_with_real_bracket_bytes() {
    // The HTML-entity corruption stalled a whole session: a `&lt;` form tokenizes to seven
    // garbage subword ids instead of one special token, silently breaking conditioning.
    for token in [
        special_tokens::IM_START,
        special_tokens::IM_END,
        special_tokens::IMAGE_PAD,
        special_tokens::VISION_START,
        special_tokens::VISION_END,
    ] {
        let bytes = token.as_bytes();
        assert_eq!(bytes[0], 0x3C, "{token} must open with a real '<'");
        assert_eq!(
            bytes[bytes.len() - 1],
            0x3E,
            "{token} must close with a real '>'"
        );
        assert!(
            !token.contains('&'),
            "{token} must not contain an HTML entity"
        );
    }
}

#[test]
fn should_normalize_an_empty_prompt_to_a_single_space() {
    assert_eq!(normalize_empty_prompt(""), " ");
    assert_eq!(normalize_empty_prompt("Romeo"), "Romeo");
    assert_eq!(normalize_empty_prompt(" "), " ");
    let rendered = render_t2i_prompt_template("");
    let expected = format!(
        "{im_start}user\n {im_end}",
        im_start = special_tokens::IM_START,
        im_end = special_tokens::IM_END
    );
    assert!(
        rendered.contains(&expected),
        "empty prompt must render as a single space; expected {expected}, got: {rendered}"
    );
}

#[test]
fn should_render_t2i_template_with_special_tokens_and_prompt() {
    let rendered = render_t2i_prompt_template("Romeo");
    assert!(rendered.starts_with(&format!("{}system\n", special_tokens::IM_START)));
    assert!(rendered.contains(QWEN_IMAGE_21_SYS_PROMPT));
    assert!(rendered.contains(&format!(
        "{}user\nRomeo{}",
        special_tokens::IM_START,
        special_tokens::IM_END
    )));
    assert!(rendered.ends_with(&format!("{}assistant\n", special_tokens::IM_START)));
    assert_eq!(rendered.matches(special_tokens::IM_START).count(), 3);
    assert_eq!(rendered.matches(special_tokens::IM_END).count(), 2);
    assert!(
        !rendered.contains('&'),
        "no HTML entities may appear in the template"
    );
}

#[test]
fn should_render_ti2i_template_with_one_vision_block_by_default() {
    let rendered = render_ti2i_prompt_template("Juliet");
    let expected_single = format!(
        "{im_start}user\n{block}Juliet",
        im_start = special_tokens::IM_START,
        block = expected_image_block(1)
    );
    assert!(
        rendered.contains(&expected_single),
        "expected substring: {expected_single}\nactual template: {rendered}"
    );
    assert!(rendered.contains(special_tokens::VISION_START));
    assert!(rendered.contains(special_tokens::IMAGE_PAD));
    assert!(rendered.contains(special_tokens::VISION_END));
    assert!(rendered.contains("Juliet"));
    assert!(
        !rendered.contains('&'),
        "no HTML entities may appear in the template"
    );
}

#[test]
fn should_render_space_separated_vision_blocks_for_each_reference_image() {
    // The reference inserts one block per image, separating second-and-later blocks from
    // the previous one with a single space, then the user prompt follows.
    let rendered = render_ti2i_prompt_template_with_image_count(
        "Juliet",
        NonZeroUsize::new(3).expect("3 is non-zero"),
    );
    let expected_blocks = format!(
        "{} {} {}",
        expected_image_block(1),
        expected_image_block(2),
        expected_image_block(3)
    );
    let expected_prefix = format!(
        "{im_start}user\n{image_blocks}Juliet",
        im_start = special_tokens::IM_START,
        image_blocks = expected_blocks
    );
    assert!(
        rendered.contains(&expected_prefix),
        "expected prefix: {expected_prefix}\nactual template: {rendered}"
    );
    assert_eq!(rendered.matches(special_tokens::IMAGE_PAD).count(), 3);
}

// ---- Validation paths (typed rejections) --------------------------------------------------

#[test]
fn should_return_ok_for_empty_batch() {
    let output = build_prompt_conditioning(&[], &[], 0, 0).expect("empty batch is valid");
    assert_eq!(output.batch_size(), 0);
    assert_eq!(output.seq_len(), 0);
}

#[test]
fn should_reject_mismatched_batch_lengths() {
    let error = build_prompt_conditioning(&[vec![1]], &[], 0, 0)
        .expect_err("batch length mismatch must fail");
    assert!(matches!(
        error,
        TextConditioningError::InputMaskLengthMismatch
    ));
}

#[test]
fn should_reject_mismatched_sample_lengths() {
    let error = build_prompt_conditioning(&[vec![1, 2]], &[vec![1, 0, 1]], 0, 0)
        .expect_err("sample length mismatch must fail");
    assert!(matches!(error, TextConditioningError::SampleLengthMismatch));
}

#[test]
fn should_reject_drop_idx_exceeding_valid_tokens() {
    let error = build_prompt_conditioning(&[vec![1, 2, 3, 4]], &[vec![1, 1, 1, 1]], 5, 0)
        .expect_err("drop_idx beyond valid count must fail");
    match error {
        TextConditioningError::DropIndexExceedsValidCount {
            sample_index,
            valid_count,
            drop_idx,
        } => {
            assert_eq!(sample_index, 0);
            assert_eq!(valid_count, 4);
            assert_eq!(drop_idx, 5);
        }
        other => panic!("expected DropIndexExceedsValidCount, got {other:?}"),
    }
}

#[test]
fn should_drop_everything_when_drop_idx_equals_valid_count() {
    // A kept length of 0 is legal: the sample contributes an all-padding row.
    let output = build_prompt_conditioning(
        &[vec![7, 8, 9], vec![1, 2, 3, 4]],
        &[vec![1, 1, 1], vec![1, 1, 1, 1]],
        3,
        0,
    )
    .expect("drop_idx equal to valid count is allowed");
    assert_eq!(output.seq_len(), 1);
    assert_eq!(output.condition_ids, vec![vec![0], vec![4]]);
    assert_eq!(output.encoder_attention_mask, vec![vec![0], vec![1]]);
    assert_eq!(output.image_pad_mask, vec![vec![0], vec![0]]);
}
