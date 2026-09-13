//! Parity contracts for the `qwen4_exp` sparse-attention indexer and the
//! attention over its selected keys.
//!
//! The user-visible outcome under test: production selection matches the
//! oracle's explicit full scoring plus host top-k under the causal mask,
//! including ties and budget edges, and the sparse attention over the
//! gathered keys matches explicit attention restricted to exactly those
//! keys. A gather that drops a selected key or a softmax over the wrong
//! axis fails here rather than degrading quality silently.

use astronomical_model_serving::{
    PerformanceAttribution, Qwen4ExpSelectionPlan, select_keys, sparse_attention,
};

use crate::direct_mlx::qwen4_exp::{
    DeterministicValues, assert_f32_close, f32_array, oracle_test_runtime,
};

/// Host-side causal top-k: descending by score, ties toward the lower
/// index, future keys masked out entirely.
fn host_causal_top_k(scores: &[f64], token_count: usize, budget: usize) -> Vec<Vec<usize>> {
    let mut selection = Vec::with_capacity(token_count);
    for query in 0..token_count {
        let mut visible: Vec<usize> = (0..=query).collect();
        visible.sort_by(|left, right| {
            scores[query * token_count + *right]
                .partial_cmp(&scores[query * token_count + *left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        });
        visible.truncate(budget);
        selection.push(visible);
    }
    selection
}

/// Host-side explicit attention restricted to a selected key set.
#[allow(clippy::too_many_arguments)]
fn host_selected_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    selection: &[Vec<usize>],
    token_count: usize,
    head_count: usize,
    key_value_head_count: usize,
    head_dim: usize,
) -> Vec<f64> {
    let group_size = head_count / key_value_head_count;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0_f64; token_count * head_count * head_dim];
    for token in 0..token_count {
        for head in 0..head_count {
            let key_value_head = head / group_size;
            let selected = &selection[token];
            let scored: Vec<(usize, f64)> = selected
                .iter()
                .map(|&key_token| {
                    let mut dot = 0.0_f64;
                    for dimension in 0..head_dim {
                        dot += queries[(token * head_count + head) * head_dim + dimension] as f64
                            * keys[(key_token * key_value_head_count + key_value_head) * head_dim
                                + dimension] as f64;
                    }
                    (key_token, dot * scale)
                })
                .collect();
            let maximum = scored
                .iter()
                .map(|(_, score)| *score)
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scored
                .iter()
                .map(|(_, score)| (score - maximum).exp())
                .collect();
            let weight_sum: f64 = weights.iter().sum();
            for dimension in 0..head_dim {
                let mut accumulated = 0.0_f64;
                for (position, key_token) in selected.iter().enumerate() {
                    accumulated += weights[position]
                        * values[(key_token * key_value_head_count + key_value_head) * head_dim
                            + dimension] as f64;
                }
                output[(token * head_count + head) * head_dim + dimension] =
                    accumulated / weight_sum;
            }
        }
    }
    output
}

#[tokio::test]
async fn should_match_causal_selection_against_the_host_reference() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x7E8F);
    for (token_count, budget) in [(8_usize, 4_usize), (12, 5), (6, 32), (5, 0), (9, 4)] {
        let scores = values.vec(token_count * token_count, 1.0);
        let scores_f64: Vec<f64> = scores.iter().map(|value| *value as f64).collect();
        let score_array = f32_array(&runtime, &scores, &[token_count as i32, token_count as i32])
            .expect("scores construct");
        let mut attribution = PerformanceAttribution::disabled();
        let selected = select_keys(
            &runtime,
            &score_array,
            &Qwen4ExpSelectionPlan::from_configuration(budget as u32),
            &mut attribution,
        )
        .expect("selection should run");
        selected.evaluate().expect("selection should evaluate");
        let selected_values = selected.to_vec_u32().expect("selection copies");
        let host = host_causal_top_k(&scores_f64, token_count, budget);
        let selected_count = budget.min(token_count);
        for query in 0..token_count {
            let production: Vec<usize> = selected_values
                [query * selected_count..(query + 1) * selected_count]
                .iter()
                .map(|index| *index as usize)
                .collect();
            // The visible prefix must match the host reference exactly; any
            // remaining slots are the documented index-zero padding for
            // queries that can see fewer keys than the budget.
            let visible_prefix = host[query].len().min(selected_count);
            assert_eq!(
                &production[..visible_prefix],
                &host[query][..visible_prefix],
                "query {query} must select the host reference set on its visible prefix (budget {budget})"
            );
            for position in visible_prefix..selected_count {
                assert_eq!(
                    production[position], 0,
                    "query {query} padding at {position} must substitute index zero"
                );
            }
        }
    }
}

#[tokio::test]
async fn should_never_select_a_future_key_even_when_the_indexer_scores_it_highly() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let token_count = 6_usize;
    // Every score is tiny except the future keys, which are enormous: the
    // causal mask must dominate selection regardless of indexer enthusiasm.
    let mut scores = vec![0.001_f32; token_count * token_count];
    for query in 0..token_count {
        for key in (query + 1)..token_count {
            scores[query * token_count + key] = 1.0e9;
        }
    }
    let score_array = f32_array(&runtime, &scores, &[token_count as i32, token_count as i32])
        .expect("scores construct");
    let mut attribution = PerformanceAttribution::disabled();
    let selected = select_keys(
        &runtime,
        &score_array,
        &Qwen4ExpSelectionPlan::from_configuration(4),
        &mut attribution,
    )
    .expect("selection should run");
    selected.evaluate().expect("selection should evaluate");
    let selected_values = selected.to_vec_u32().expect("selection copies");
    for query in 0..token_count {
        let visible_count = (query + 1).min(4);
        for position in 0..visible_count {
            let key = selected_values[query * 4 + position] as usize;
            assert!(
                key <= query,
                "query {query} selected future key {key}; the causal mask failed"
            );
        }
    }
}

#[tokio::test]
async fn should_match_sparse_attention_against_explicit_attention_on_the_selected_keys() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let mut values = DeterministicValues::new(0x8F90);
    let token_count = 9_usize;
    let head_count = 4_usize;
    let key_value_head_count = 2_usize;
    let head_dim = 6_usize;
    let budget = 4_usize;
    let queries = values.vec(token_count * head_count * head_dim, 1.0);
    let keys = values.vec(token_count * key_value_head_count * head_dim, 1.0);
    let raw_values = values.vec(token_count * key_value_head_count * head_dim, 1.0);
    let scores = values.vec(token_count * token_count, 1.0);
    let scores_f64: Vec<f64> = scores.iter().map(|value| *value as f64).collect();
    let selection = host_causal_top_k(&scores_f64, token_count, budget);
    // Pad every row to the uniform selected count with index zero, matching
    // the production contract, so the host reference attends over exactly
    // the same slots the production path receives.
    let selected_count = budget.min(token_count);
    let selection: Vec<Vec<usize>> = selection
        .into_iter()
        .map(|mut row| {
            row.resize(selected_count, 0);
            row
        })
        .collect();

    let query_array = f32_array(
        &runtime,
        &queries,
        &[token_count as i32, head_count as i32, head_dim as i32],
    )
    .expect("queries construct");
    let key_array = f32_array(
        &runtime,
        &keys,
        &[
            token_count as i32,
            key_value_head_count as i32,
            head_dim as i32,
        ],
    )
    .expect("keys construct");
    let value_array = f32_array(
        &runtime,
        &raw_values,
        &[
            token_count as i32,
            key_value_head_count as i32,
            head_dim as i32,
        ],
    )
    .expect("values construct");
    let selected_count = budget.min(token_count);
    let mut flat_selection = Vec::with_capacity(token_count * selected_count);
    for query in 0..token_count {
        for position in 0..selected_count {
            flat_selection.push(selection[query][position] as u32);
        }
    }
    let selection_array = runtime
        .array_from_u32(
            &flat_selection,
            &[token_count as i32, selected_count as i32],
        )
        .expect("selection constructs");
    let mut attribution = PerformanceAttribution::disabled();
    let attended = sparse_attention(
        &runtime,
        &query_array,
        &key_array,
        &value_array,
        &selection_array,
        &mut attribution,
    )
    .expect("sparse attention should run");
    attended
        .evaluate()
        .expect("sparse attention should evaluate");
    let production = attended.to_vec_f32().expect("attention copies");
    let host = host_selected_attention(
        &queries,
        &keys,
        &raw_values,
        &selection,
        token_count,
        head_count,
        key_value_head_count,
        head_dim,
    );
    assert_f32_close(
        &production,
        &host,
        1.0e-3,
        "sparse attention over selected keys must match explicit attention",
    );
}

#[tokio::test]
async fn should_reject_a_non_square_score_matrix() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let scores = vec![0.0_f32; 12];
    let score_array = f32_array(&runtime, &scores, &[3, 4]).expect("scores construct");
    let mut attribution = PerformanceAttribution::disabled();
    let error = select_keys(
        &runtime,
        &score_array,
        &Qwen4ExpSelectionPlan::from_configuration(2),
        &mut attribution,
    )
    .expect_err("a non-square score matrix must fail");
    assert!(
        error
            .to_string()
            .contains("must be [token_count, token_count]"),
        "the error should name the square-shape rule: {error}"
    );
}
