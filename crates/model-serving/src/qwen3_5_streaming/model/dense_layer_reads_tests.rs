//! Contracts for planning a dense layer's bounded loads: one load per shard,
//! file-ordered intervals, merged neighbours, and a header that names every
//! tensor over a contiguous virtual payload.

use std::collections::BTreeMap;

use super::dense_layer_reads::{DenseLayerTensorRange, dense_layer_shard_reads};

fn tensor_range(
    canonical_name: &str,
    file_start_offset: u64,
    byte_count: u64,
) -> DenseLayerTensorRange {
    DenseLayerTensorRange {
        canonical_name: canonical_name.to_owned(),
        stored_dtype: "U32".to_owned(),
        shape: vec![(byte_count / 4) as usize],
        file_start_offset,
        file_end_offset: file_start_offset + byte_count,
    }
}

fn header_json(synthetic_header_bytes: &[u8]) -> serde_json::Value {
    let header_length = u64::from_le_bytes(
        synthetic_header_bytes[..8]
            .try_into()
            .expect("the header should start with its length"),
    ) as usize;
    assert_eq!(synthetic_header_bytes.len(), 8 + header_length);
    serde_json::from_slice(&synthetic_header_bytes[8..]).expect("the header should be JSON")
}

#[test]
fn should_merge_neighbouring_tensors_and_split_at_gaps_in_file_order() {
    let shard_reads = dense_layer_shard_reads(BTreeMap::from([(
        0,
        vec![
            tensor_range(
                "language_model.model.layers.5.mlp.up_proj.weight",
                9_000,
                400,
            ),
            tensor_range(
                "language_model.model.layers.5.mlp.down_proj.weight",
                1_000,
                200,
            ),
            tensor_range(
                "language_model.model.layers.5.mlp.gate_proj.weight",
                1_200,
                100,
            ),
        ],
    )]));

    assert_eq!(shard_reads.len(), 1);
    let shard_read = &shard_reads[0];
    assert_eq!(shard_read.payload_bytes, 700);
    assert_eq!(shard_read.read_intervals.len(), 2);
    assert_eq!(shard_read.read_intervals[0].source_file_offset, 1_000);
    assert_eq!(shard_read.read_intervals[0].source_byte_count, 300);
    assert_eq!(shard_read.read_intervals[0].virtual_payload_offset, 0);
    assert_eq!(shard_read.read_intervals[1].source_file_offset, 9_000);
    assert_eq!(shard_read.read_intervals[1].virtual_payload_offset, 300);
}

#[test]
fn should_name_every_tensor_over_a_contiguous_virtual_payload() {
    let shard_reads = dense_layer_shard_reads(BTreeMap::from([(
        0,
        vec![
            tensor_range(
                "language_model.model.layers.0.mlp.up_proj.weight",
                5_000,
                400,
            ),
            tensor_range(
                "language_model.model.layers.0.input_layernorm.weight",
                100,
                8,
            ),
        ],
    )]));

    assert_eq!(
        shard_reads[0].tensor_names,
        [
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
        ],
        "the read lists its tensors in file order"
    );
    let header = header_json(&shard_reads[0].synthetic_header_bytes);
    assert_eq!(
        header["language_model.model.layers.0.input_layernorm.weight"]["data_offsets"],
        serde_json::json!([0, 8])
    );
    assert_eq!(
        header["language_model.model.layers.0.mlp.up_proj.weight"]["data_offsets"],
        serde_json::json!([8, 408])
    );
    assert_eq!(
        header["language_model.model.layers.0.mlp.up_proj.weight"]["dtype"],
        "U32"
    );
}

#[test]
fn should_plan_one_load_per_shard_that_holds_part_of_the_layer() {
    let shard_reads = dense_layer_shard_reads(BTreeMap::from([
        (
            2,
            vec![tensor_range(
                "language_model.model.layers.9.mlp.down_proj.weight",
                0,
                64,
            )],
        ),
        (
            1,
            vec![tensor_range(
                "language_model.model.layers.9.mlp.up_proj.weight",
                32,
                64,
            )],
        ),
    ]));

    assert_eq!(shard_reads.len(), 2);
    assert_eq!(shard_reads[0].shard_position, 1);
    assert_eq!(shard_reads[1].shard_position, 2);
    assert_eq!(shard_reads[1].read_intervals[0].source_file_offset, 0);
}

#[test]
fn should_plan_no_loads_for_a_layer_with_no_tensors() {
    assert!(dense_layer_shard_reads(BTreeMap::new()).is_empty());
}

#[test]
fn should_plan_an_empty_read_for_a_reversed_range_instead_of_underflowing() {
    let mut reversed_range =
        tensor_range("language_model.model.layers.1.mlp.up_proj.weight", 500, 0);
    reversed_range.file_end_offset = 100;

    let shard_reads = dense_layer_shard_reads(BTreeMap::from([(0, vec![reversed_range])]));

    assert_eq!(shard_reads[0].payload_bytes, 0);
}
