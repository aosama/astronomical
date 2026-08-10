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
    assert!(decoder_cache_layout.has_sequence_state());
    assert!(decoder_cache_layout.has_boundary_state());
    assert_eq!(
        decoder_cache_layout
            .sequence_state_payload_byte_count_per_token()
            .expect("sequence payload bytes should fit usize"),
        32
    );
    assert_eq!(
        decoder_cache_layout
            .maximum_sequence_tensor_payload_byte_count(128)
            .expect("the largest sequence tensor should fit usize"),
        2_048
    );
    assert_eq!(
        decoder_cache_layout
            .persistence_alignment_token_count()
            .expect("the append-only alignment should fit usize"),
        256
    );
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
fn should_derive_the_least_common_persistence_alignment_for_mixed_attention_growth() {
    let decoder_cache_layout = DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "first.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "first.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            6,
        ),
        DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "second.keys",
                DecoderCacheTensorDtype::Float32,
                vec![1, 0, 2],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "second.values",
                DecoderCacheTensorDtype::Float32,
                vec![1, 0, 2],
                1,
            ),
            8,
        ),
    ])
    .expect("mixed append-only growth should form a valid layout");

    assert_eq!(
        decoder_cache_layout
            .persistence_alignment_token_count()
            .expect("the least common alignment should fit usize"),
        24
    );
}

#[test]
fn should_reject_persistence_alignment_overflow() {
    let decoder_cache_layout = DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "first.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "first.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            usize::MAX,
        ),
        DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "second.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "second.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            usize::MAX - 1,
        ),
    ])
    .expect("the individual append-only growth values are valid");

    assert!(matches!(
        decoder_cache_layout.persistence_alignment_token_count(),
        Err(DecoderCacheLayoutError::PersistenceAlignmentTokenCountOverflow)
    ));
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
