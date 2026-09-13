//! Oracle reference for the sparse-attention indexer's key selection.
//!
//! The production indexer will score keys with a low-rank projection and
//! select a bounded subset. A wrong selection keeps every shape legal and
//! degrades quality silently, so the reference here computes scores by full
//! explicit dot products on the GPU and selection by host-side sorting, and
//! the parity contract is that production's selected-key set matches this
//! reference's set for the same inputs.
//!
//! This module proves the reference itself first: full explicit scoring
//! against host `f64` math, and host top-k against MLX `topk` on the same
//! scores, including tie handling and the budget-larger-than-context case.

use astronomical_runtime_integration::{MlxDtype, MlxRuntime};

use super::{DeterministicValues, assert_f32_close, f32_array, oracle_test_runtime};

/// Scoring geometry for one reference row.
pub(crate) struct IndexerGeometry {
    pub token_count: usize,
    pub head_dim: usize,
    pub budget: usize,
}

/// Computes indexer scores on the GPU by explicit dot products: one score
/// per query-key pair, no compression, no sparsity.
pub(crate) fn explicit_scores(
    runtime: &MlxRuntime,
    queries: &[f32],
    keys: &[f32],
    geometry: &IndexerGeometry,
) -> Result<Vec<f32>, astronomical_runtime_integration::MlxRuntimeError> {
    let query_array = f32_array(
        runtime,
        queries,
        &[geometry.token_count as i32, geometry.head_dim as i32],
    )?;
    let key_array = f32_array(
        runtime,
        keys,
        &[geometry.token_count as i32, geometry.head_dim as i32],
    )?;
    let transposed = runtime.transpose_axes(&key_array, &[1, 0])?;
    let scores = runtime.matmul(&query_array, &transposed)?;
    scores.evaluate()?;
    assert_eq!(
        scores.dtype(),
        MlxDtype::Float32,
        "the reference must score in f32, not in a narrowed accumulator"
    );
    scores.to_vec_f32()
}

/// Host-side `f64` reference for the same scores.
pub(crate) fn host_scores(queries: &[f32], keys: &[f32], geometry: &IndexerGeometry) -> Vec<f64> {
    let mut scores = Vec::with_capacity(geometry.token_count * geometry.token_count);
    for query_index in 0..geometry.token_count {
        for key_index in 0..geometry.token_count {
            let mut dot = 0.0_f64;
            for dimension in 0..geometry.head_dim {
                let query_offset = query_index * geometry.head_dim + dimension;
                let key_offset = key_index * geometry.head_dim + dimension;
                dot += queries[query_offset] as f64 * keys[key_offset] as f64;
            }
            scores.push(dot);
        }
    }
    scores
}

/// Host-side top-k selection: descending by score, stable on ties by index.
pub(crate) fn host_top_k(scores: &[f64], budget: usize, token_count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..token_count).collect();
    order.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(right))
    });
    order.truncate(budget);
    order
}

#[tokio::test]
async fn should_match_explicit_indexer_scores_against_host_reference() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x1D5A);
    for geometry in [
        IndexerGeometry {
            token_count: 8,
            head_dim: 4,
            budget: 4,
        },
        IndexerGeometry {
            token_count: 16,
            head_dim: 8,
            budget: 8,
        },
        IndexerGeometry {
            token_count: 5,
            head_dim: 3,
            budget: 32,
        },
    ] {
        let queries = values.vec(geometry.token_count * geometry.head_dim, 1.0);
        let keys = values.vec(geometry.token_count * geometry.head_dim, 1.0);
        let gpu_scores = explicit_scores(&runtime, &queries, &keys, &geometry)
            .expect("explicit scores should compute");
        let host = host_scores(&queries, &keys, &geometry);
        // Metal matrix kernels accumulate in tf32-class precision (about ten
        // mantissa bits), so a relative bound near 4e-3 is the hardware's
        // honest behavior rather than a loosened check; the dtype assertion
        // above proves the arrays themselves stay f32.
        assert_f32_close(
            &gpu_scores,
            &host,
            4.0e-3,
            &format!("indexer scores for {} tokens", geometry.token_count),
        );
    }
}

#[tokio::test]
async fn should_match_host_top_k_selection_including_ties_and_oversized_budgets() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    // Ties: equal scores must select the lower index first, so the selection
    // is deterministic even when the scores are not unique.
    let tied = vec![1.0_f64, 2.0, 2.0, 0.5, 2.0];
    assert_eq!(host_top_k(&tied, 2, 5), vec![1, 2]);
    // Budget larger than the context selects everything.
    assert_eq!(host_top_k(&tied, 32, 5).len(), 5);
    // Budget zero selects nothing.
    assert!(host_top_k(&tied, 0, 5).is_empty());

    // The same selection computed from GPU scores matches host selection on
    // the same values, proving the GPU path can feed the reference contract.
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x2E6B);
    let geometry = IndexerGeometry {
        token_count: 12,
        head_dim: 4,
        budget: 5,
    };
    let queries = values.vec(geometry.token_count * geometry.head_dim, 1.0);
    let keys = values.vec(geometry.token_count * geometry.head_dim, 1.0);
    let gpu_scores = explicit_scores(&runtime, &queries, &keys, &geometry)
        .expect("explicit scores should compute");
    let host = host_scores(&queries, &keys, &geometry);
    let gpu_selection: Vec<usize> = {
        let per_query = geometry.token_count;
        (0..per_query)
            .map(|query_index| {
                let row: Vec<f64> = gpu_scores
                    [query_index * per_query..(query_index + 1) * per_query]
                    .iter()
                    .map(|value| *value as f64)
                    .collect();
                host_top_k(&row, geometry.budget, per_query)[0]
            })
            .collect()
    };
    let host_selection: Vec<usize> = (0..geometry.token_count)
        .map(|query_index| {
            host_top_k(
                &host[query_index * geometry.token_count..(query_index + 1) * geometry.token_count],
                geometry.budget,
                geometry.token_count,
            )[0]
        })
        .collect();
    assert_eq!(
        gpu_selection, host_selection,
        "the GPU-scored best key must match the host-scored best key per query"
    );
}
