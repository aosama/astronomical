use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    FIXTURE_VOCAB_SIZE, remove_fixture_directory, write_fixture_pack,
};
use crate::prism_hadamard_qwen35::artifacts::{FoldedPackValidator, ValidatedFoldedPack};
use text_generation::generation_loop::serving_artifact::TextServingArtifact;

/// The fixture pack's one full-attention layer stores 2 key-value heads of
/// 128 Float16 elements for its keys and again for its values.
const FIXTURE_KEY_VALUE_BYTES_PER_CONTEXT_TOKEN: usize = 2 * 2 * 128 * 2;

fn with_validated_fixture_pack(check_validated_pack: fn(&ValidatedFoldedPack)) {
    let fixture_directory = write_fixture_pack(|_| {});
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    check_validated_pack(&validated_pack);
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_answer_the_identity_and_limits_the_prism_engine_reads() {
    with_validated_fixture_pack(|validated_pack| {
        let serving_artifact: &dyn TextServingArtifact = validated_pack;
        assert_eq!(serving_artifact.model_id(), validated_pack.model_id());
        assert_eq!(serving_artifact.revision(), validated_pack.revision());
        assert_eq!(
            serving_artifact.total_payload_bytes(),
            validated_pack.total_payload_bytes()
        );
        assert_eq!(serving_artifact.shard_count(), 1);
        assert!(
            serving_artifact
                .persistent_visual_embedding_model_contract()
                .is_none()
        );
        assert_eq!(serving_artifact.maximum_position_count(), 4096);
        assert_eq!(serving_artifact.vocabulary_size(), FIXTURE_VOCAB_SIZE);
        assert_eq!(serving_artifact.end_of_sequence_token_ids(), &[248_044]);
    });
}

#[test]
fn should_reserve_full_attention_keys_and_values_per_context_token() {
    with_validated_fixture_pack(|validated_pack| {
        assert_eq!(
            validated_pack.context_memory_reservation_bytes(1),
            Some(FIXTURE_KEY_VALUE_BYTES_PER_CONTEXT_TOKEN)
        );
        assert_eq!(
            validated_pack.context_memory_reservation_bytes(4096),
            Some(4096 * FIXTURE_KEY_VALUE_BYTES_PER_CONTEXT_TOKEN)
        );
        assert_eq!(validated_pack.context_memory_reservation_bytes(0), Some(0));
    });
}

#[test]
fn should_refuse_a_context_reservation_that_overflows() {
    with_validated_fixture_pack(|validated_pack| {
        assert_eq!(
            validated_pack.context_memory_reservation_bytes(usize::MAX),
            None
        );
    });
}
