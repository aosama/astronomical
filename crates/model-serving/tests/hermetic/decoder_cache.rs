use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheLayoutError, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout,
};

#[test]
fn should_accept_a_mixed_architecture_neutral_decoder_cache_layout() {
    let decoder_cache_layout = DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "linear.convolution",
            DecoderCacheTensorDtype::Float16,
            vec![1, 3, 8],
        )),
        DecoderCacheLayerLayout::composite(vec![
            DecoderCacheLayerLayout::append_only_attention(
                DecoderCacheTensorLayout::sequence(
                    "attention.keys",
                    DecoderCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                DecoderCacheTensorLayout::sequence(
                    "attention.values",
                    DecoderCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                256,
            ),
            DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
                "linear.recurrent",
                DecoderCacheTensorDtype::Float32,
                vec![1, 2, 4, 4],
            )),
        ]),
    ])
    .expect("a mixed decoder-cache layout should be valid");

    assert_eq!(decoder_cache_layout.layer_count(), 2);
    assert_eq!(decoder_cache_layout.sequence_tensor_count(), 2);
    assert_eq!(decoder_cache_layout.boundary_tensor_count(), 2);
    assert_eq!(
        decoder_cache_layout
            .boundary_snapshot_payload_byte_count()
            .expect("fixed boundary tensor payload bytes should fit usize"),
        176
    );
    assert_eq!(
        decoder_cache_layout
            .sequence_tensor_layouts()
            .into_iter()
            .map(|tensor_layout| tensor_layout.persistent_tensor_name())
            .collect::<Vec<_>>(),
        vec![
            "layer_1_attention.keys".to_owned(),
            "layer_1_attention.values".to_owned(),
        ]
    );
    assert_eq!(
        decoder_cache_layout
            .boundary_tensor_layouts()
            .into_iter()
            .map(|tensor_layout| tensor_layout.persistent_tensor_name())
            .collect::<Vec<_>>(),
        vec![
            "layer_0_linear.convolution".to_owned(),
            "layer_1_linear.recurrent".to_owned(),
        ]
    );
}

#[test]
fn should_reject_duplicate_qualified_tensor_roles_across_one_layer() {
    let rejection = DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::composite(vec![
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "state",
            DecoderCacheTensorDtype::Float32,
            vec![1, 4],
        )),
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "state",
            DecoderCacheTensorDtype::Float32,
            vec![1, 4],
        )),
    ])]);

    assert!(matches!(
        rejection,
        Err(DecoderCacheLayoutError::DuplicateTensorRole { layer_index: 0, .. })
    ));
}

#[test]
fn should_reject_a_sequence_axis_outside_the_tensor_rank() {
    let rejection = DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
        DecoderCacheTensorLayout::sequence(
            "attention.keys",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 2, 0, 4],
            4,
        ),
        DecoderCacheTensorLayout::sequence(
            "attention.values",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 2, 0, 4],
            2,
        ),
        256,
    )]);

    assert!(matches!(
        rejection,
        Err(DecoderCacheLayoutError::SequenceAxisOutsideTensorRank { layer_index: 0, .. })
    ));
}
