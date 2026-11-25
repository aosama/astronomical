use crate::qwen3_5_core::artifacts::qwen3_5_language_tensor_profiles;
use crate::qwen3_5_core::qwen4_exp_config_test_support::frozen_flash_next_config;
use crate::{TensorDtype, TensorProfile};

use super::is_ngram_table_shard_tensor_name;

fn published_profiles() -> Vec<TensorProfile> {
    qwen3_5_language_tensor_profiles(&frozen_flash_next_config())
        .expect("the published Flash-Next configuration builds its inventory")
}

fn profile_named<'profiles>(
    profiles: &'profiles [TensorProfile],
    name: &str,
) -> &'profiles TensorProfile {
    profiles
        .iter()
        .find(|profile| profile.name == name)
        .unwrap_or_else(|| panic!("{name} is missing from the inventory"))
}

#[test]
fn should_list_exactly_as_many_language_tensors_as_the_published_index() {
    // The published 4-bit index names 3,434 `language_model.*` tensors.
    assert_eq!(published_profiles().len(), 3_434);
}

#[test]
fn should_expect_the_published_shapes_of_the_flash_next_only_tensors() {
    let profiles = published_profiles();
    let expectations: [(&str, TensorDtype, Vec<usize>); 8] = [
        (
            "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            TensorDtype::UInt32,
            vec![4, 1280],
        ),
        (
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
            TensorDtype::UInt32,
            vec![10240, 40],
        ),
        (
            "language_model.model.layers.1.ple.conv1d.weight",
            TensorDtype::ModelFloat,
            vec![10240, 4, 1],
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.127.weight",
            TensorDtype::UInt32,
            vec![2_500_012, 20],
        ),
        (
            "language_model.model.layers.1.ple.ple_embedding.layer_multipliers",
            TensorDtype::Int64,
            vec![3],
        ),
        (
            "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight",
            TensorDtype::UInt32,
            vec![640, 320],
        ),
        (
            "language_model.model.layers.0.mlp.gate.weight",
            TensorDtype::UInt32,
            // 8-bit, group 64: 2,560 inputs pack into 640 words.
            vec![512, 640],
        ),
        (
            "language_model.model.layers.3.self_attn.indexer.k_layernorm.weight",
            TensorDtype::ModelFloat,
            vec![128],
        ),
    ];
    for (name, dtype, shape) in expectations {
        let profile = profile_named(&profiles, name);
        assert_eq!(profile.dtype, dtype, "{name}");
        assert_eq!(profile.shape, shape, "{name}");
    }
}

#[test]
fn should_carry_no_per_layer_norms_and_no_final_norm() {
    let profiles = published_profiles();
    assert!(
        !profiles
            .iter()
            .any(|profile| profile.name.ends_with("_layernorm.weight")
                && !profile.name.contains(".indexer.")),
        "the hyper-connections replace the input and post-attention norms"
    );
    assert!(
        !profiles
            .iter()
            .any(|profile| profile.name == "language_model.model.norm.weight"),
        "the final hyper-connection mixer replaces the final norm"
    );
}

#[test]
fn should_put_the_ngram_table_only_on_the_ngram_layer() {
    let profiles = published_profiles();
    let ngram_shard_names = profiles
        .iter()
        .filter(|profile| is_ngram_table_shard_tensor_name(&profile.name))
        .map(|profile| profile.name.as_str())
        .collect::<Vec<_>>();
    // 128 shards, each a packed weight with its scales and biases.
    assert_eq!(ngram_shard_names.len(), 384);
    assert!(
        ngram_shard_names
            .iter()
            .all(|name| name.starts_with("language_model.model.layers.1.ple."))
    );
}

#[test]
fn should_not_take_an_mtp_or_vision_tensor_for_a_shard() {
    assert!(!is_ngram_table_shard_tensor_name(
        "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
    ));
    assert!(!is_ngram_table_shard_tensor_name(
        "language_model.model.layers.1.ple.key_proj.weight"
    ));
}
