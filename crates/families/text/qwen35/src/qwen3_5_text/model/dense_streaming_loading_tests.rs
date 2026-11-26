//! Beside-source coverage for what a model with no dense streaming leaves
//! unchanged. The branches that read layer payload sizes need a planned
//! streamer, which the dense layer streaming lane covers.

use super::{dense_streaming_geometry, embedding_residency_of};
use crate::qwen3_5_core::model_math::weights::EmbeddingResidency;

#[test]
fn should_leave_the_core_whole_and_reserve_no_workspace_without_a_streamer() {
    let geometry = dense_streaming_geometry(7_000_000, None);

    assert_eq!(geometry.model_core_payload_bytes, 7_000_000);
    assert_eq!(geometry.dense_streaming_workspace_bytes, 0);
}

#[test]
fn should_keep_an_empty_core_empty_without_a_streamer() {
    let geometry = dense_streaming_geometry(0, None);

    assert_eq!(geometry.model_core_payload_bytes, 0);
    assert_eq!(geometry.dense_streaming_workspace_bytes, 0);
}

#[test]
fn should_keep_the_embedding_table_resident_when_no_rows_are_read_from_disk() {
    assert_eq!(embedding_residency_of(None), EmbeddingResidency::Resident);
}
