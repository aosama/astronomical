use std::fs::{self, File};

use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION, PersistentVisualEmbeddingFileHeader,
    PersistentVisualEmbeddingKey,
};
use serde_json::json;

use crate::common::qwen3_5_moe::persistent_visual_embedding_model_contract;

#[test]
fn should_accept_a_well_formed_bfloat16_visual_embedding_file() {
    let temporary_directory = tempfile::tempdir().expect("the test should create a directory");
    let encoded_image_sha256 = [7_u8; 32];
    let visual_embedding_key = PersistentVisualEmbeddingKey::for_image(
        encoded_image_sha256,
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    );
    let visual_embedding_file_path = temporary_directory.path().join(format!(
        "{}.safetensors",
        hex_encode(visual_embedding_key.visual_embedding_hash())
    ));
    write_visual_embedding_file(&visual_embedding_file_path, encoded_image_sha256, 2);

    let visual_embedding_file =
        File::open(&visual_embedding_file_path).expect("the visual file should open");
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();
    let visual_embedding_file_header = PersistentVisualEmbeddingFileHeader::read_from_file(
        &visual_embedding_file,
        &visual_embedding_file_path,
        &persistent_visual_embedding_model_contract,
    )
    .expect("the valid visual file header should be accepted");

    assert_eq!(
        visual_embedding_file_header.format_version(),
        PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION
    );
    assert_eq!(
        visual_embedding_file_header.model_id(),
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID
    );
    assert_eq!(
        visual_embedding_file_header.model_revision(),
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION
    );
    assert_eq!(
        visual_embedding_file_header.encoded_image_sha256(),
        encoded_image_sha256
    );
    assert_eq!(visual_embedding_file_header.visual_token_count(), 2);
}

#[test]
fn should_reject_a_visual_file_whose_filename_does_not_match_its_metadata_digest() {
    let temporary_directory = tempfile::tempdir().expect("the test should create a directory");
    let encoded_image_sha256 = [7_u8; 32];
    let mismatched_file_path = temporary_directory
        .path()
        .join(format!("{}.safetensors", "00".repeat(32)));
    write_visual_embedding_file(&mismatched_file_path, encoded_image_sha256, 2);

    let visual_embedding_file =
        File::open(&mismatched_file_path).expect("the visual file should open");
    let persistent_visual_embedding_model_contract = persistent_visual_embedding_model_contract();
    let header_result = PersistentVisualEmbeddingFileHeader::read_from_file(
        &visual_embedding_file,
        &mismatched_file_path,
        &persistent_visual_embedding_model_contract,
    );

    assert!(
        header_result.is_err(),
        "a visual file must be content-addressed by its metadata digest"
    );
}

fn write_visual_embedding_file(
    visual_embedding_file_path: &std::path::Path,
    encoded_image_sha256: [u8; 32],
    visual_token_count: usize,
) {
    let expected_payload_byte_count = visual_token_count * 2_048 * 2;
    let mut header_json = serde_json::to_vec(&json!({
        "__metadata__": {
            "format_version": PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION,
            "model_id": ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
            "model_revision": ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
            "encoded_image_sha256": hex_encode(encoded_image_sha256),
            "visual_token_count": visual_token_count.to_string(),
        },
        "visual_embeddings": {
            "dtype": "BF16",
            "shape": [visual_token_count, 2_048],
            "data_offsets": [0, expected_payload_byte_count],
        },
    }))
    .expect("the test visual header should serialize");
    header_json.extend(std::iter::repeat_n(b' ', (8 - header_json.len() % 8) % 8));
    let mut visual_file_bytes = (header_json.len() as u64).to_le_bytes().to_vec();
    visual_file_bytes.extend(header_json);
    visual_file_bytes.extend(std::iter::repeat_n(0_u8, expected_payload_byte_count));
    fs::write(visual_embedding_file_path, visual_file_bytes)
        .expect("the test visual file should be written");
}

fn hex_encode(digest_bytes: [u8; 32]) -> String {
    digest_bytes
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect()
}
