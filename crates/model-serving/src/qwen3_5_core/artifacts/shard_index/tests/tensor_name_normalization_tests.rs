//! Tests for the shard-index tensor-name normalization that accepts newer
//! unified Hugging Face checkpoints (`model.language_model.*`,
//! `model.visual.*`) alongside the canonical executable spellings
//! (`language_model.model.*`, `vision_tower.*`).

use std::collections::BTreeMap;

use serde_json::json;

use crate::TensorDtype;
use crate::TensorProfile;

use super::*;

fn synthetic_unquantized_language_tensor_profile(tensor_name: &str) -> TensorProfile {
    tensor_spec::qwen3_5_tensor_profile(tensor_name.to_owned(), TensorDtype::ModelFloat, vec![8, 4])
}

fn serialize_index_bytes(weight_map: BTreeMap<String, String>) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "metadata": { "total_size": 64 },
        "weight_map": weight_map,
    }))
    .expect("the synthetic shard index should serialize")
}

#[test]
fn should_map_the_unified_language_prefix_to_the_canonical_language_layout() {
    assert_eq!(
        canonical_model_tensor_name("model.language_model.embed_tokens.weight"),
        Some("language_model.model.embed_tokens.weight".to_owned())
    );
    assert_eq!(
        canonical_model_tensor_name("model.language_model.layers.5.mlp.gate_proj.weight"),
        Some("language_model.model.layers.5.mlp.gate_proj.weight".to_owned())
    );
}

#[test]
fn should_map_the_unified_visual_prefix_to_the_canonical_vision_tower_layout() {
    assert_eq!(
        canonical_model_tensor_name("model.visual.patch_embed.proj.weight"),
        Some("vision_tower.patch_embed.proj.weight".to_owned())
    );
    assert_eq!(
        canonical_model_tensor_name("model.visual.blocks.3.attn.qkv.bias"),
        Some("vision_tower.blocks.3.attn.qkv.bias".to_owned())
    );
}

#[test]
fn should_pass_canonical_language_vision_and_mtp_names_through_unchanged() {
    assert_eq!(
        canonical_model_tensor_name("language_model.model.embed_tokens.weight"),
        Some("language_model.model.embed_tokens.weight".to_owned())
    );
    assert_eq!(
        canonical_model_tensor_name("vision_tower.patch_embed.proj.weight"),
        Some("vision_tower.patch_embed.proj.weight".to_owned())
    );
    assert_eq!(
        canonical_model_tensor_name("mtp.fc.weight"),
        Some("mtp.fc.weight".to_owned())
    );
}

#[test]
fn should_reject_tensor_names_outside_the_executable_model_prefixes() {
    assert_eq!(
        canonical_model_tensor_name("model.embed_tokens.weight"),
        None
    );
    assert_eq!(canonical_model_tensor_name("lm_head.weight"), None);
    assert_eq!(canonical_model_tensor_name(""), None);
}

#[test]
fn should_treat_prefix_boundaries_exactly() {
    // Missing the trailing dot of a unified prefix is a foreign name, not a
    // partial match.
    assert_eq!(canonical_model_tensor_name("model.language_model"), None);
    assert_eq!(canonical_model_tensor_name("model.visual"), None);
    assert_eq!(canonical_model_tensor_name("language_model"), None);
    assert_eq!(canonical_model_tensor_name("mtp"), None);
    // A different tower number never matches the visual prefix.
    assert_eq!(canonical_model_tensor_name("model.visual2.weight"), None);
    // The degenerate empty-suffix form still normalizes deterministically.
    assert_eq!(
        canonical_model_tensor_name("model.language_model."),
        Some("language_model.model.".to_owned())
    );
}

#[test]
fn should_normalize_a_unified_naming_index_into_canonical_buckets() {
    let mut weight_map = BTreeMap::new();
    weight_map.insert(
        "model.language_model.embed_tokens.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    weight_map.insert(
        "model.visual.patch_embed.proj.weight".to_owned(),
        "vision-only.safetensors".to_owned(),
    );
    weight_map.insert(
        "mtp.fc.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    weight_map.insert(
        "trainer.random_state".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    let index_bytes = serialize_index_bytes(weight_map);
    let language_tensor_profiles = [synthetic_unquantized_language_tensor_profile(
        "language_model.model.embed_tokens.weight",
    )];

    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the unified-naming shard index should parse into canonical buckets");

    assert_eq!(
        shard_index
            .language_tensor_name_to_shard_file_name()
            .keys()
            .next()
            .map(String::as_str),
        Some("language_model.model.embed_tokens.weight")
    );
    assert_eq!(
        shard_index
            .vision_tensor_name_to_shard_file_name()
            .keys()
            .next()
            .map(String::as_str),
        Some("vision_tower.patch_embed.proj.weight")
    );
    // The vision-only file stays a sidecar even though it holds a unified
    // spelling.
    assert_eq!(
        shard_index.vision_sidecar_file_names(),
        &["vision-only.safetensors".to_owned()]
    );
    assert_eq!(
        shard_index.model_shard_file_names(),
        &["model-00001.safetensors".to_owned()]
    );
    assert_eq!(
        shard_index
            .auxiliary_tensor_name_to_shard_file_name()
            .get("mtp.fc.weight")
            .map(String::as_str),
        Some("model-00001.safetensors")
    );
}

#[test]
fn should_resolve_canonical_names_to_stored_unified_spellings() {
    let mut weight_map = BTreeMap::new();
    weight_map.insert(
        "model.language_model.embed_tokens.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    weight_map.insert(
        "mtp.fc.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    let index_bytes = serialize_index_bytes(weight_map);
    let language_tensor_profiles = [synthetic_unquantized_language_tensor_profile(
        "language_model.model.embed_tokens.weight",
    )];

    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles)
        .expect("the unified-naming shard index should parse");

    // Renamed tensors resolve to their stored unified spelling.
    assert_eq!(
        shard_index.stored_tensor_name_for("language_model.model.embed_tokens.weight"),
        "model.language_model.embed_tokens.weight"
    );
    // Canonical-named tensors resolve to themselves.
    assert_eq!(
        shard_index.stored_tensor_name_for("mtp.fc.weight"),
        "mtp.fc.weight"
    );
}

#[test]
fn should_skip_foreign_tensor_prefixes_silently() {
    let mut weight_map = BTreeMap::new();
    weight_map.insert(
        "trainer.random_state".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    let index_bytes = serialize_index_bytes(weight_map);

    let shard_index = Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &[]);

    assert!(shard_index.is_ok());
    let shard_index = shard_index.expect("foreign prefixes should be skipped, not rejected");
    assert!(
        shard_index
            .language_tensor_name_to_shard_file_name()
            .is_empty()
    );
    assert!(
        shard_index
            .vision_tensor_name_to_shard_file_name()
            .is_empty()
    );
    assert!(
        shard_index
            .auxiliary_tensor_name_to_shard_file_name()
            .is_empty()
    );
    assert!(shard_index.model_shard_file_names().is_empty());
}

#[test]
fn should_reject_conflicting_stored_spellings_for_one_canonical_tensor() {
    let mut weight_map = BTreeMap::new();
    weight_map.insert(
        "model.language_model.embed_tokens.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    weight_map.insert(
        "language_model.model.embed_tokens.weight".to_owned(),
        "model-00001.safetensors".to_owned(),
    );
    let index_bytes = serialize_index_bytes(weight_map);
    let language_tensor_profiles = [synthetic_unquantized_language_tensor_profile(
        "language_model.model.embed_tokens.weight",
    )];

    match Qwen3_5ShardIndex::from_json_bytes(&index_bytes, &language_tensor_profiles) {
        Err(Qwen3_5ArtifactError::ConflictingCanonicalTensorName {
            canonical_tensor_name,
            existing_stored_tensor_name,
            conflicting_stored_tensor_name,
        }) => {
            assert_eq!(
                canonical_tensor_name, "language_model.model.embed_tokens.weight",
                "the conflict must be reported under the canonical name"
            );
            // The index weight-map iteration order decides which spelling is
            // reported as existing, so the test asserts the pair rather than
            // the order.
            let reported_stored_spellings =
                [existing_stored_tensor_name, conflicting_stored_tensor_name];
            assert!(
                reported_stored_spellings
                    .contains(&"model.language_model.embed_tokens.weight".to_owned())
            );
            assert!(
                reported_stored_spellings
                    .contains(&"language_model.model.embed_tokens.weight".to_owned())
            );
        }
        other => panic!("a conflicting second spelling must be rejected, got: {other:?}"),
    }
}
