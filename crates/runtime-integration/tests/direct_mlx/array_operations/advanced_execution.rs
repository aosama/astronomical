use crate::common::runtime_test_support::runtime;

#[test]
fn should_apply_unmasked_scaled_dot_product_attention() {
    let runtime = runtime();
    let queries = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the query tensor should be valid");
    let keys = runtime
        .array_from_f32(&[0.0, 0.0], &[1, 1, 2, 1])
        .expect("the key tensor should be valid");
    let values = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2, 1])
        .expect("the value tensor should be valid");
    let attention = runtime
        .scaled_dot_product_attention(&queries, &keys, &values, 1.0)
        .expect("scaled dot-product attention should build a valid graph");
    assert_eq!(attention.shape(), vec![1, 1, 1, 1]);
    assert_eq!(
        attention
            .to_vec_f32()
            .expect("the attention output should evaluate as float32"),
        vec![3.0]
    );
}

#[test]
fn should_apply_causal_scaled_dot_product_attention() {
    let runtime = runtime();
    let queries = runtime
        .array_from_f32(&[1.0, 1.0], &[1, 1, 2, 1])
        .expect("the query tensor should be valid");
    let keys = runtime
        .array_from_f32(&[0.0, 0.0], &[1, 1, 2, 1])
        .expect("the key tensor should be valid");
    let values = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2, 1])
        .expect("the value tensor should be valid");
    let attention = runtime
        .causal_scaled_dot_product_attention(&queries, &keys, &values, 1.0)
        .expect("causal scaled dot-product attention should build a valid graph");
    assert_eq!(attention.shape(), vec![1, 1, 2, 1]);
    assert_eq!(
        attention
            .to_vec_f32()
            .expect("the causal attention output should evaluate as float32"),
        vec![2.0, 3.0]
    );
}

#[test]
fn should_sample_categorical_token_with_a_fixed_key_seed() {
    let runtime = runtime();
    let logits = runtime
        .array_from_f32(&[0.0, 0.0, 0.0], &[1, 3])
        .expect("the categorical logits should be valid");
    let sampled_token = runtime
        .categorical_sample(&logits, -1, 2)
        .expect("categorical sampling should build a valid graph");
    assert_eq!(
        sampled_token
            .item_u32()
            .expect("sampled token should read as u32"),
        1
    );
}

#[test]
fn should_match_the_upstream_seeded_categorical_key_sequence() {
    let runtime = runtime();
    let logits = runtime
        .array_from_f32(&[0.0, 1.0, 2.0], &[1, 3])
        .expect("the categorical logits should be valid");
    let mut random_state = runtime
        .random_key(1_234)
        .expect("the upstream seed should create one random state");
    let mut sampled_token_ids = Vec::new();
    for _sample_number in 0..6 {
        let (next_random_state, sample_key) = runtime
            .split_random_key(&random_state)
            .expect("each categorical draw should split the random state");
        let sampled_token = runtime
            .categorical_sample_with_key(&logits, -1, &sample_key)
            .expect("categorical sampling should use the split sample key");
        sampled_token_ids.push(
            sampled_token
                .item_u32()
                .expect("the sampled token should evaluate as uint32"),
        );
        random_state = next_random_state;
    }
    // MLX v0.32.1 intentionally uses inverse cumulative-distribution sampling
    // for this singleton-batch shape, so its fixed-key oracle differs from v0.32.0.
    assert_eq!(sampled_token_ids, vec![2, 2, 0, 2, 2, 2]);
}

#[test]
fn should_async_evaluate_dependent_arrays_without_losing_their_results() {
    let runtime = runtime();
    let source_values = runtime
        .array_from_f32(&[1.0, 2.0, 3.0], &[3])
        .expect("the source values should be valid");
    let doubled_values = runtime
        .multiply_scalar(&source_values, 2.0)
        .expect("the first dependent graph should be valid");
    runtime
        .async_eval_arrays(&[&doubled_values])
        .expect("the first graph should submit asynchronously");
    let quadrupled_values = runtime
        .multiply_scalar(&doubled_values, 2.0)
        .expect("the second dependent graph should be valid");
    runtime
        .async_eval_arrays(&[&quadrupled_values])
        .expect("the dependent graph should submit asynchronously");
    assert_eq!(
        quadrupled_values
            .to_vec_f32()
            .expect("the asynchronously evaluated values should remain readable"),
        vec![4.0, 8.0, 12.0]
    );
}

#[test]
fn should_replace_only_the_selected_static_slice() {
    let runtime = runtime();
    let cache_storage = runtime
        .array_from_f32(&[0.0; 8], &[2, 4])
        .expect("the cache storage should be valid");
    let cache_update = runtime
        .array_from_f32(&[5.0, 7.0], &[1, 2])
        .expect("the cache update should be valid");
    let updated_cache = runtime
        .slice_update(&cache_storage, &cache_update, &[1, 1], &[2, 3], &[1, 1])
        .expect("the static slice update should build a valid graph");
    assert_eq!(
        updated_cache
            .to_vec_f32()
            .expect("the updated cache should evaluate as float32"),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 7.0, 0.0]
    );
}
