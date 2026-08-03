use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentVisualEmbeddingKey,
};

#[test]
fn should_produce_the_same_visual_file_identity_for_the_same_image_digest() {
    let encoded_image_sha256 = [7_u8; 32];

    let first_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        encoded_image_sha256,
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let second_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        encoded_image_sha256,
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );

    assert_eq!(
        first_visual_embedding_key.visual_embedding_hash(),
        second_visual_embedding_key.visual_embedding_hash()
    );
}

#[test]
fn should_isolate_visual_file_identity_for_different_image_digests() {
    let first_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [7_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let second_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [8_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );

    assert_ne!(
        first_visual_embedding_key.visual_embedding_hash(),
        second_visual_embedding_key.visual_embedding_hash()
    );
}

#[test]
fn should_not_reuse_format_one_visual_embedding_key_after_domain_seed_rename() {
    let current_visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        [7_u8; 32],
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );

    assert_ne!(
        current_visual_embedding_key.visual_embedding_hash(),
        format_one_visual_embedding_key([7_u8; 32]),
        "domain seed and format version rename must produce a disjoint visual embedding namespace"
    );
}

/// Reproduces the format-1 visual embedding hash using the old domain seed.
/// This is intentional regression evidence: the old domain bytes appear
/// only here as historical comparison data, not in any production path.
fn format_one_visual_embedding_key(encoded_image_sha256: [u8; 32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update((b"astronomical-ornith-visual-embedding".len() as u64).to_be_bytes());
    digest.update(b"astronomical-ornith-visual-embedding");
    digest.update((b"1".len() as u64).to_be_bytes());
    digest.update(b"1");
    digest.update((ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.len() as u64).to_be_bytes());
    digest.update(ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.as_bytes());
    digest.update((ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.len() as u64).to_be_bytes());
    digest.update(ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.as_bytes());
    digest.update((32_u64).to_be_bytes());
    digest.update(encoded_image_sha256);
    digest.finalize().into()
}
