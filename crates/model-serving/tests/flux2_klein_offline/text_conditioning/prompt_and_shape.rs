use super::batch::{
    FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH, FLUX2_KLEIN_PAD_TOKEN_ID, prepare_token_rows,
};
use super::prompt::Flux2KleinPromptRenderer;
use super::tokenizer::Flux2KleinTokenizer;

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[test]
fn should_render_one_romeo_and_juliet_user_message_with_thinking_disabled() {
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");

    let rendered_prompt = Flux2KleinPromptRenderer::render_user_prompt(&source_excerpt);

    assert_eq!(
        rendered_prompt,
        format!(
            "<|im_start|>user\n{source_excerpt}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
}

#[test]
fn should_right_pad_romeo_and_juliet_tokens_and_emit_the_attention_mask() {
    let source_token_ids = romeo_and_juliet_token_ids(7);

    let prepared_batch = prepare_token_rows(vec![source_token_ids.clone()])
        .expect("the Romeo and Juliet token row should prepare");

    assert_eq!(prepared_batch.batch_size(), 1);
    assert_eq!(
        prepared_batch.sequence_length(),
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH
    );
    assert_eq!(&prepared_batch.token_ids()[..7], source_token_ids);
    assert!(
        prepared_batch.token_ids()[7..]
            .iter()
            .all(|token_id| *token_id == FLUX2_KLEIN_PAD_TOKEN_ID)
    );
    assert_eq!(&prepared_batch.attention_mask()[..7], &[1; 7]);
    assert!(
        prepared_batch.attention_mask()[7..]
            .iter()
            .all(|mask| *mask == 0)
    );
}

#[test]
fn should_load_tokenizer_json_from_the_retained_descriptor_and_prepare_romeo() {
    let fixture_directory =
        tempfile::tempdir().expect("the retained-tokenizer fixture directory should be created");
    let tokenizer_path = fixture_directory.path().join("tokenizer.json");
    std::fs::write(&tokenizer_path, word_level_tokenizer_json())
        .expect("the retained tokenizer descriptor should be written");
    let retained_sidecars = std::collections::BTreeMap::from([(
        "tokenizer/tokenizer.json".to_owned(),
        std::fs::File::open(tokenizer_path).expect("the retained tokenizer descriptor should open"),
    )]);
    let source_excerpt = ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ");

    let tokenizer = Flux2KleinTokenizer::from_retained_sidecars(&retained_sidecars)
        .expect("tokenizer JSON should load from retained descriptor bytes");
    let prepared_batch = tokenizer
        .prepare_rendered_prompts(&Flux2KleinPromptRenderer::render_user_prompts(&[
            source_excerpt,
        ]))
        .expect("the retained tokenizer should prepare the Romeo prompt");

    assert_eq!(prepared_batch.batch_size(), 1);
    assert_eq!(prepared_batch.token_ids().len(), 512);
    assert_eq!(prepared_batch.attention_mask().len(), 512);
    assert!(prepared_batch.attention_mask().contains(&1));
}

#[test]
fn should_right_truncate_romeo_and_juliet_tokens_to_exactly_512() {
    let source_token_ids = romeo_and_juliet_token_ids(640);

    let prepared_batch = prepare_token_rows(vec![source_token_ids.clone()])
        .expect("the long Romeo and Juliet token row should prepare");

    assert_eq!(
        prepared_batch.token_ids(),
        &source_token_ids[..FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH]
    );
    assert!(
        prepared_batch
            .attention_mask()
            .iter()
            .all(|mask| *mask == 1)
    );
}

#[test]
fn should_reject_an_empty_prompt_batch_before_native_execution() {
    prepare_token_rows(Vec::new()).expect_err("an empty image prompt batch must be rejected");
}

fn romeo_and_juliet_token_ids(token_count: usize) -> Vec<u32> {
    ROMEO_AND_JULIET_SOURCE
        .bytes()
        .cycle()
        .take(token_count)
        .map(|source_byte| u32::from(source_byte) + 1_000)
        .collect()
}

fn word_level_tokenizer_json() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "Romeo": 1, "Juliet": 2},
            "unk_token": "<unk>"
        }
    }))
    .expect("the synthetic retained tokenizer JSON should serialize")
}
