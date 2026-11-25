use super::NgramTableGeometry;
use crate::qwen3_5_core::qwen4_exp_config_test_support::frozen_flash_next_config;

fn published_geometry() -> NgramTableGeometry {
    let architecture = frozen_flash_next_config()
        .qwen4_exp_architecture()
        .expect("Flash-Next carries its additions");
    NgramTableGeometry::for_architecture(&architecture)
}

#[test]
fn should_draw_the_head_table_sizes_the_published_checkpoint_stores() {
    // Read from `ple.ple_embedding.ngram_heads_vocab_sizes` of the 4-bit
    // checkpoint; the table sizes are part of the trained hash, not tunable.
    let checkpoint_head_sizes = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    let geometry = published_geometry();
    assert_eq!(geometry.head_vocabulary_sizes(), checkpoint_head_sizes);
    assert_eq!(geometry.head_row_offsets()[1], 20_000_003);
    assert_eq!(geometry.head_row_offsets()[15], 300_001_275);
}

#[test]
fn should_pad_the_table_and_split_it_into_the_published_shard_shapes() {
    let geometry = published_geometry();
    // The published shards each store 2,500,012 rows.
    assert_eq!(geometry.padded_row_count(), 320_001_536);
    assert_eq!(geometry.shard_count(), 128);
    assert_eq!(geometry.shard_row_count(0), 2_500_012);
    assert_eq!(geometry.shard_row_count(127), 2_500_012);
}

#[test]
fn should_map_rows_at_shard_boundaries_to_the_right_shard() {
    let geometry = published_geometry();
    assert_eq!(geometry.shard_and_local_row(0), Some((0, 0)));
    assert_eq!(
        geometry.shard_and_local_row(2_500_011),
        Some((0, 2_500_011))
    );
    assert_eq!(geometry.shard_and_local_row(2_500_012), Some((1, 0)));
    assert_eq!(
        geometry.shard_and_local_row(320_001_535),
        Some((127, 2_500_011))
    );
}

#[test]
fn should_refuse_a_row_past_the_padded_table() {
    assert_eq!(published_geometry().shard_and_local_row(320_001_536), None);
}

#[test]
fn should_give_the_first_shards_the_extra_rows_when_rows_do_not_divide_evenly() {
    let first_rows = super::shard_first_rows(10, 4);
    // 10 rows over 4 shards: 3, 3, 2, 2.
    assert_eq!(first_rows, vec![0, 3, 6, 8]);
}
