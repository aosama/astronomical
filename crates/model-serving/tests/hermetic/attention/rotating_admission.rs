use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout, rotating_committed_token_count,
    rotating_prefill_transient_token_count,
};

#[test]
fn should_charge_window_plus_chunk_minus_one_for_rotating_prefill() {
    for (window_size, chunk_tokens, expected_peak_tokens) in [
        (8_u32, 1_u32, 8_u32),
        (8, 3, 10),
        (64, 16, 79),
        (512, 128, 639),
        (4, 0, 4),
    ] {
        assert_eq!(
            rotating_prefill_transient_token_count(window_size, chunk_tokens)
                .expect("positive geometry should be valid"),
            expected_peak_tokens
        );
    }
}

#[test]
fn should_bound_committed_tokens_to_the_window() {
    assert_eq!(rotating_committed_token_count(8, 3), 3);
    assert_eq!(rotating_committed_token_count(8, 40), 8);
}

#[test]
fn should_validate_and_persist_rotating_boundary_geometry() {
    let zero_window = DecoderCacheLayout::new(vec![rotating_layout(0)]);
    assert_eq!(
        zero_window,
        Err(DecoderCacheLayoutError::ZeroRotatingWindowSize { layer_index: 0 })
    );

    let layout = DecoderCacheLayout::new(vec![rotating_layout(8)])
        .expect("a positive rotating window should validate");
    assert_eq!(layout.sequence_tensor_count(), 0);
    assert_eq!(layout.boundary_tensor_count(), 4);
    let persisted_names = layout
        .boundary_tensor_layouts()
        .into_iter()
        .map(|tensor_layout| tensor_layout.persistent_tensor_name())
        .collect::<Vec<_>>();
    assert_eq!(
        persisted_names,
        vec![
            "layer_0_attention.keys",
            "layer_0_attention.values",
            "layer_0_attention.absolute_position",
            "layer_0_attention.ring_write_index",
        ]
    );
}

#[test]
fn should_reject_zero_window_and_overflowing_transient_geometry() {
    assert!(rotating_prefill_transient_token_count(0, 4).is_err());
    assert!(rotating_prefill_transient_token_count(u32::MAX, 2).is_err());
}

fn rotating_layout(window_size: usize) -> DecoderCacheLayerLayout {
    DecoderCacheLayerLayout::rotating_window_attention(
        DecoderCacheTensorLayout::fixed(
            "attention.keys",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 2, 8, 4],
        ),
        DecoderCacheTensorLayout::fixed(
            "attention.values",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 2, 8, 4],
        ),
        window_size,
    )
}
