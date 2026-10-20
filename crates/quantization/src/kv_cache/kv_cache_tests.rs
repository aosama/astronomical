//! Unit tests for the key-value cache allocator.

use super::*;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

fn result(layer_index: usize, kl_at_four: f64) -> KvCacheSensitivityResult {
    KvCacheSensitivityResult {
        layer_index,
        layer_name: format!("model.layers.{layer_index}.self_attn"),
        kl_divergence_by_bit_width: BTreeMap::from([(bits(4), kl_at_four), (bits(8), 0.0)]),
        key_value_head_count: 8,
        head_dimension: 128,
        cache_bytes_per_token: 2.0 * 8.0 * 128.0 * 2.0,
    }
}

#[test]
#[ignore = "pending implementation of optimize_kv_cache"]
fn should_protect_the_most_sensitive_layer_first() {
    // Layer 0 is 56x more sensitive than the others; at a 5.0 average over
    // four equal layers exactly one layer goes to 8 bits and it is layer 0.
    let results = vec![
        result(0, 0.56),
        result(1, 0.01),
        result(2, 0.01),
        result(3, 0.01),
    ];
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let configs = optimize_kv_cache(
        &results,
        5.0,
        &candidate_bit_widths,
        GroupSize::new(64).expect("valid"),
    )
    .expect("allocation should succeed");

    let layer_zero = configs
        .iter()
        .find(|config| config.layer_index == 0)
        .expect("layer 0 is configured");
    assert_eq!(layer_zero.bit_width, bits(8));
    assert_eq!(
        configs
            .iter()
            .filter(|config| config.bit_width == bits(8))
            .count(),
        1
    );
}

#[test]
#[ignore = "pending implementation of estimate_kv_cache_bytes"]
fn should_scale_cache_bytes_with_bit_width() {
    let results = vec![result(0, 0.1)];
    let four_bit = vec![KvCacheLayerConfig {
        layer_index: 0,
        layer_name: "model.layers.0.self_attn".to_owned(),
        bit_width: bits(4),
        group_size: GroupSize::new(64).expect("valid"),
    }];
    let eight_bit = vec![KvCacheLayerConfig {
        bit_width: bits(8),
        ..four_bit[0].clone()
    }];

    assert!(
        estimate_kv_cache_bytes(&four_bit, &results, 4096)
            < estimate_kv_cache_bytes(&eight_bit, &results, 4096)
    );
}
