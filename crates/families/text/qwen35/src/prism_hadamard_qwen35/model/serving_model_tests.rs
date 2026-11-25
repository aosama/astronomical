use serial_test::file_serial;

use super::super::fixture_model_test_support::load_fixture_text_model;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::FIXTURE_VOCAB_SIZE;
use text_generation::model::TextServingModel;

#[test]
#[file_serial]
fn should_answer_the_limits_and_layout_the_prism_engine_uses() {
    let mut fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    assert_eq!(
        TextServingModel::vocabulary_size(text_model),
        FIXTURE_VOCAB_SIZE
    );
    assert_eq!(TextServingModel::maximum_position_count(text_model), 4096);
    assert!(std::ptr::eq(
        TextServingModel::astro_decoder_state_cache_layout(text_model),
        &text_model.astro_decoder_state_cache_layout
    ));
    assert!(std::ptr::eq(
        TextServingModel::runtime(text_model),
        &text_model.runtime
    ));
    assert!(std::ptr::eq(
        TextServingModel::compiled_elementwise_graphs(text_model),
        &text_model.compiled_elementwise_graphs
    ));
    // Edge: a text-only pack has no tower, and releasing one changes nothing.
    assert_eq!(
        TextServingModel::vision_tower_resident_payload_bytes(text_model),
        0
    );
    TextServingModel::release_vision_tower_after_image_request(&mut fixture.text_model);
    assert_eq!(
        TextServingModel::vision_tower_resident_payload_bytes(&fixture.text_model),
        0
    );
}

#[test]
#[file_serial]
fn should_choose_the_same_greedy_token_as_prism_sampling() {
    let fixture = load_fixture_text_model();
    let text_model = &fixture.text_model;
    // A strictly increasing score row puts the highest score on the last id.
    let ascending_scores: Vec<f32> = (0..FIXTURE_VOCAB_SIZE)
        .map(|token_id| token_id as f32)
        .collect();
    let final_position_logits = text_model
        .runtime
        .array_from_f32(&ascending_scores, &[1, 1, FIXTURE_VOCAB_SIZE as i32])
        .expect("the score row should build");

    let shared_choice =
        TextServingModel::select_highest_logit_token(text_model, &final_position_logits)
            .expect("the greedy choice should build")
            .item_u32()
            .expect("the greedy choice should evaluate");
    let prism_choice = text_model
        .greedy_token_lazily(&final_position_logits)
        .expect("Prism's greedy choice should build")
        .item_u32()
        .expect("Prism's greedy choice should evaluate");

    assert_eq!(shared_choice, prism_choice);
    assert_eq!(shared_choice, FIXTURE_VOCAB_SIZE - 1);
}
