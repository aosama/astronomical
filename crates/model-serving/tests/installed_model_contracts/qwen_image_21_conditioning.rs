//! End-to-end validation of the Qwen-Image-2.1 text-conditioning package against the
//! real artifact tokenizer (the `tokenizers` crate reading the artifact's
//! `processor/tokenizer.json`).
//!
//! The hermetic suite (`qwen_image_21_hermetic`) checks the packaging algebra against a
//! committed oracle fixture. These ignored journeys run the real tokenizer and prove that
//! the Rust-rendered templates byte-match the diffusers reference *and* that the artifact
//! tokenizer encodes the Qwen chat special tokens as single ids. Without the real
//! tokenizer, a rendered template with a corrupted HTML-entity bracket (`&lt;` instead of
//! `&lt;`) would still pass the algebra but tokenize to seven garbage subword ids; this is
//! the only way to guard against that corruption end to end.
//!
//! Resolves the artifact through `ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY` so no
//! developer path is hardcoded; the artifact is not registered in the Development model
//! directory, so it cannot be found by ordinary discovery.

use astronomical_model_serving::{
    build_prompt_conditioning, render_t2i_prompt_template, render_ti2i_prompt_template,
    render_ti2i_prompt_template_with_image_count, special_tokens,
};
use std::num::NonZeroUsize;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

use crate::common::qwen_image_21::artifact_directory;

const ORACLE_FIXTURE_JSON: &str =
    include_str!("../fixtures/qwen_image_21/conditioning_oracle.json");

/// The Qwen chat special tokens that must each encode to exactly one id in the real tokenizer.
const SPECIAL_TOKEN_IDS: [(u32, &str); 5] = [
    (151_644, special_tokens::IM_START),
    (151_645, special_tokens::IM_END),
    (151_652, special_tokens::VISION_START),
    (151_653, special_tokens::VISION_END),
    (151_655, special_tokens::IMAGE_PAD),
];

/// Loads the real artifact tokenizer with the standalone `tokenizers` crate.
fn artifact_tokenizer() -> Tokenizer {
    let processor_path = artifact_directory().join("processor");
    assert!(
        processor_path.join("tokenizer.json").exists(),
        "expected the Qwen-Image-2.1 artifact tokenizer at {}",
        processor_path.join("tokenizer.json").display()
    );
    Tokenizer::from_file(processor_path.join("tokenizer.json"))
        .expect("the Qwen-Image-2.1 tokenizer.json must parse")
}

/// A decoded oracle case: prompts (aligned by index with the row arrays), the image count, and
/// the row-parallel raw + packaged matrices.
struct OraclePrompt {
    prompts: Vec<String>,
    image_count: usize,
    max_seq_len: usize,
    raw_input_ids: Vec<Vec<u32>>,
    raw_attention_mask: Vec<Vec<u32>>,
    expected_condition_ids: Vec<Vec<u32>>,
    expected_attention_mask: Vec<Vec<u8>>,
    expected_image_pad_mask: Vec<Vec<u8>>,
}

fn oracle_case(value: &serde_json::Value) -> OraclePrompt {
    let read_rows = |field: &str| -> Vec<Vec<u32>> {
        let array = value[field]
            .as_array()
            .unwrap_or_else(|| panic!("oracle {field} must be an array"));
        array
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or_else(|| panic!("oracle row must be an array"))
                    .iter()
                    .map(|element| element.as_u64().unwrap_or(0) as u32)
                    .collect()
            })
            .collect()
    };
    let read_u8_rows = |field: &str| -> Vec<Vec<u8>> {
        let array = value[field]
            .as_array()
            .unwrap_or_else(|| panic!("oracle {field} must be an array"));
        array
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or_else(|| panic!("oracle row must be an array"))
                    .iter()
                    .map(|element| element.as_u64().unwrap_or(0) as u8)
                    .collect()
            })
            .collect()
    };
    let prompts: Vec<String> = value["prompts"]
        .as_array()
        .expect("oracle prompts must be an array")
        .iter()
        .map(|element| element.as_str().unwrap_or("").to_owned())
        .collect();
    let raw_input_ids = read_rows("raw_input_ids");
    assert_eq!(
        prompts.len(),
        raw_input_ids.len(),
        "oracle prompt count must match the raw row count"
    );
    OraclePrompt {
        prompts,
        image_count: value["image_count"].as_u64().unwrap_or(0) as usize,
        max_seq_len: value["max_seq_len"].as_u64().unwrap_or(0) as usize,
        raw_input_ids,
        raw_attention_mask: read_rows("raw_attention_mask"),
        expected_condition_ids: read_rows("expected_condition_ids"),
        expected_attention_mask: read_u8_rows("expected_attention_mask"),
        expected_image_pad_mask: read_u8_rows("expected_image_pad_mask"),
    }
}

struct OracleFixture {
    drop_idx: usize,
    image_pad_token_id: u32,
    pad_token_id: u32,
    t2i: OraclePrompt,
    ti2i: OraclePrompt,
}

/// Parses the committed oracle fixture once per test binary.
fn oracle_fixture() -> &'static OracleFixture {
    static FIXTURE: OnceLock<OracleFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let value: serde_json::Value = serde_json::from_str(ORACLE_FIXTURE_JSON)
            .expect("the oracle fixture must parse as JSON");
        OracleFixture {
            drop_idx: value["drop_idx"].as_u64().unwrap_or(0) as usize,
            image_pad_token_id: value["image_pad_token_id"].as_u64().unwrap_or(0) as u32,
            pad_token_id: value["pad_token_id"].as_u64().unwrap_or(0) as u32,
            t2i: oracle_case(&value["t2i"]),
            ti2i: oracle_case(&value["ti2i"]),
        }
    })
}

/// Encodes a prompt with the real artifact tokenizer, the same way the oracle tokenized the
/// rendered template (no padding, so the result is the unpadded valid token sequence).
fn encode_template(tokenizer: &Tokenizer, rendered_prompt: &str) -> Vec<u32> {
    tokenizer
        .encode(rendered_prompt, false)
        .expect("the rendered template should encode")
        .get_ids()
        .to_vec()
}

/// The oracle's valid (mask==1) tokens for a sample, i.e. the unpadded encoded row.
fn oracle_valid_tokens(raw_ids: &[u32], raw_mask: &[u32]) -> Vec<u32> {
    (0..raw_ids.len())
        .filter(|&index| raw_mask[index] > 0)
        .map(|index| raw_ids[index])
        .collect()
}

#[test]
#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
fn should_encode_chat_special_tokens_as_single_ids() {
    let tokenizer = artifact_tokenizer();

    // The single strongest guard: each Qwen chat special token must encode to exactly one id
    // in the real tokenizer. A corrupted HTML-entity bracket (`&lt;`) would expand into
    // several subword ids, so this fails loudly if the compiled template leaked an entity
    // form into a render.
    for (expected_id, token) in SPECIAL_TOKEN_IDS {
        let encoded = encode_template(&tokenizer, token);
        assert_eq!(
            encoded,
            vec![expected_id],
            "special token {token:?} must encode to the single id {expected_id}, got {encoded:?}"
        );
    }
}

#[test]
#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
fn should_render_and_encode_t2i_templates_bit_identically_to_the_oracle() {
    let fixture = oracle_fixture();
    let tokenizer = artifact_tokenizer();

    // The Rust-rendered template must produce exactly the unpadded oracle row for every
    // sample. If the artifact tokenizer and the oracle's transformers tokenizer disagree,
    // the whole packaging assumption is wrong, so the comparison is strict.
    for (index, prompt) in fixture.t2i.prompts.iter().enumerate() {
        let rendered = render_t2i_prompt_template(prompt);
        let encoded = encode_template(&tokenizer, &rendered);
        assert_eq!(
            encoded,
            oracle_valid_tokens(
                &fixture.t2i.raw_input_ids[index],
                &fixture.t2i.raw_attention_mask[index]
            ),
            "Rust-rendered T2I template must encode to the oracle row for prompt {index}"
        );
    }

    // Drop-index alignment: after dropping the first `drop_idx` valid tokens, the first
    // remaining token must be the user-turn `im_start` marker (151644).
    let encoded_rows: Vec<Vec<u32>> = fixture
        .t2i
        .prompts
        .iter()
        .map(|prompt| encode_template(&tokenizer, &render_t2i_prompt_template(prompt)))
        .collect();
    let encoded_masks: Vec<Vec<u32>> = encoded_rows.iter().map(|row| vec![1; row.len()]).collect();
    let packaged = build_prompt_conditioning(
        &encoded_rows,
        &encoded_masks,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the real T2I template must package");
    assert_eq!(
        packaged.condition_ids[0][0], 151_644,
        "post-drop T2I template must start with the user-turn im_start marker"
    );
}

#[test]
#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
fn should_render_and_encode_ti2i_templates_bit_identically_to_the_oracle() {
    let fixture = oracle_fixture();
    let tokenizer = artifact_tokenizer();

    for (index, prompt) in fixture.ti2i.prompts.iter().enumerate() {
        let rendered = if fixture.ti2i.image_count > 1 {
            render_ti2i_prompt_template_with_image_count(
                prompt,
                NonZeroUsize::new(fixture.ti2i.image_count).unwrap(),
            )
        } else {
            render_ti2i_prompt_template(prompt)
        };
        let encoded = encode_template(&tokenizer, &rendered);
        assert_eq!(
            encoded,
            oracle_valid_tokens(
                &fixture.ti2i.raw_input_ids[index],
                &fixture.ti2i.raw_attention_mask[index]
            ),
            "Rust-rendered ti2i template must encode to the oracle row for prompt {index}"
        );
    }

    // The ti2i template must carry exactly one reference-image slot: a single image_pad id
    // among the valid tokens, sitting between vision_start and vision_end.
    let first_valid = oracle_valid_tokens(
        &fixture.ti2i.raw_input_ids[0],
        &fixture.ti2i.raw_attention_mask[0],
    );
    let image_pad_positions: Vec<usize> = (0..first_valid.len())
        .filter(|&index| first_valid[index] == fixture.image_pad_token_id)
        .collect();
    assert_eq!(
        image_pad_positions.len(),
        1,
        "the ti2i template must carry exactly one image_pad position, found {:?}",
        image_pad_positions
    );
}

#[test]
#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
fn should_derive_the_real_drop_idx_alignment_and_packaging_from_the_tokenizer() {
    let fixture = oracle_fixture();
    let tokenizer = artifact_tokenizer();

    // The drop index the artifact derives must land between the system block and the user
    // turn for every sample: the first `drop_idx` valid tokens are the system block.
    assert_eq!(
        fixture.drop_idx, 14,
        "the reviewed artifact must drop the same 14 system tokens"
    );

    // The oracle fixture's pad id must be a real artifact vocab id, so the left-padded
    // positions the reference prepends decode to that token, not an arbitrary id.
    let vocab_ids: Vec<u32> = tokenizer.get_vocab(true).values().copied().collect();
    assert!(
        vocab_ids.contains(&fixture.pad_token_id),
        "the oracle fixture pad id {pad_id} must be a real artifact vocab id",
        pad_id = fixture.pad_token_id
    );

    // Packaging the oracle's raw (possibly left-padded) rows through the same packaging
    // algebra must reproduce the oracle packaged matrices exactly, validating drop-index
    // alignment, the image-pad mask, and the right-padding against the real artifact.
    let packaged = build_prompt_conditioning(
        &fixture.t2i.raw_input_ids,
        &fixture.t2i.raw_attention_mask,
        fixture.drop_idx,
        fixture.image_pad_token_id,
    )
    .expect("the oracle T2I rows must package");

    assert_eq!(
        packaged.condition_ids, fixture.t2i.expected_condition_ids,
        "packaged T2I ids must equal the oracle exactly"
    );
    assert_eq!(
        packaged.encoder_attention_mask, fixture.t2i.expected_attention_mask,
        "packaged T2I attention mask must equal the oracle exactly"
    );
    assert_eq!(
        packaged.image_pad_mask, fixture.t2i.expected_image_pad_mask,
        "packaged T2I image-pad mask must equal the oracle exactly"
    );

    // The packaged sequence length must match the oracle's post-drop maximum.
    assert_eq!(
        packaged.seq_len(),
        fixture.t2i.max_seq_len,
        "packaged T2I sequence length must equal the oracle max_seq_len"
    );
}
