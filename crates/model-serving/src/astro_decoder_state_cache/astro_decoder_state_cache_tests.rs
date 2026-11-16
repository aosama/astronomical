use crate::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCacheLayoutError, AstroDecoderStateCacheTensorDtype,
    AstroDecoderStateCacheTensorLayout,
};

#[test]
fn should_accept_a_mixed_architecture_neutral_astro_decoder_state_cache_layout() {
    let astro_decoder_state_cache_layout = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::recurrent_tensor(
            AstroDecoderStateCacheTensorLayout::fixed(
                "linear.convolution",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 3, 8],
            ),
        ),
        AstroDecoderStateCacheLayerLayout::composite(vec![
            AstroDecoderStateCacheLayerLayout::append_only_attention(
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.keys",
                    AstroDecoderStateCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.values",
                    AstroDecoderStateCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                256,
            ),
            AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                AstroDecoderStateCacheTensorLayout::fixed(
                    "linear.recurrent",
                    AstroDecoderStateCacheTensorDtype::Float32,
                    vec![1, 2, 4, 4],
                ),
            ),
        ]),
    ])
    .expect("a mixed astro-decoder-state-cache layout should be valid");

    assert_eq!(astro_decoder_state_cache_layout.layer_count(), 2);
    assert_eq!(astro_decoder_state_cache_layout.sequence_tensor_count(), 2);
    assert_eq!(astro_decoder_state_cache_layout.boundary_tensor_count(), 2);
    assert!(astro_decoder_state_cache_layout.has_sequence_state());
    assert!(astro_decoder_state_cache_layout.has_boundary_state());
    assert_eq!(
        astro_decoder_state_cache_layout
            .sequence_state_payload_byte_count_per_token()
            .expect("sequence payload bytes should fit usize"),
        32
    );
    assert_eq!(
        astro_decoder_state_cache_layout
            .maximum_sequence_tensor_payload_byte_count(128)
            .expect("the largest sequence tensor should fit usize"),
        2_048
    );
    assert_eq!(
        astro_decoder_state_cache_layout
            .persistence_alignment_token_count()
            .expect("the append-only alignment should fit usize"),
        256
    );
    assert_eq!(
        astro_decoder_state_cache_layout
            .boundary_snapshot_payload_byte_count()
            .expect("fixed boundary tensor payload bytes should fit usize"),
        176
    );
    assert_eq!(
        astro_decoder_state_cache_layout
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
        astro_decoder_state_cache_layout
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
fn should_bound_incremental_restore_workspace_by_one_block_or_boundary_snapshot() {
    let astro_decoder_state_cache_layout = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::recurrent_tensor(
            AstroDecoderStateCacheTensorLayout::fixed(
                "linear.convolution",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 3, 8],
            ),
        ),
        AstroDecoderStateCacheLayerLayout::composite(vec![
            AstroDecoderStateCacheLayerLayout::append_only_attention(
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.keys",
                    AstroDecoderStateCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                AstroDecoderStateCacheTensorLayout::sequence(
                    "attention.values",
                    AstroDecoderStateCacheTensorDtype::BFloat16,
                    vec![1, 2, 0, 4],
                    2,
                ),
                256,
            ),
            AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                AstroDecoderStateCacheTensorLayout::fixed(
                    "linear.recurrent",
                    AstroDecoderStateCacheTensorDtype::Float32,
                    vec![1, 2, 4, 4],
                ),
            ),
        ]),
    ])
    .expect("the mixed astro decoder state cache layout should be valid");

    assert_eq!(
        astro_decoder_state_cache_layout
            .incremental_restore_source_workspace_byte_count(2)
            .expect("the boundary snapshot source should fit"),
        176,
        "the complete boundary snapshot must remain budgeted when larger than one sequence block"
    );
    assert_eq!(
        astro_decoder_state_cache_layout
            .incremental_restore_source_workspace_byte_count(16)
            .expect("the sequence block source should fit"),
        512,
        "sequence source admission should scale with one block, not the restored prefix"
    );
}

#[test]
fn should_derive_the_least_common_persistence_alignment_for_mixed_attention_growth() {
    let astro_decoder_state_cache_layout = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "first.keys",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "first.values",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            6,
        ),
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "second.keys",
                AstroDecoderStateCacheTensorDtype::Float32,
                vec![1, 0, 2],
                1,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "second.values",
                AstroDecoderStateCacheTensorDtype::Float32,
                vec![1, 0, 2],
                1,
            ),
            8,
        ),
    ])
    .expect("mixed append-only growth should form a valid layout");

    assert_eq!(
        astro_decoder_state_cache_layout
            .persistence_alignment_token_count()
            .expect("the least common alignment should fit usize"),
        24
    );
}

#[test]
fn should_reject_persistence_alignment_overflow() {
    let astro_decoder_state_cache_layout = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "first.keys",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "first.values",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            usize::MAX,
        ),
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "second.keys",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "second.values",
                AstroDecoderStateCacheTensorDtype::Float16,
                vec![1, 0],
                1,
            ),
            usize::MAX - 1,
        ),
    ])
    .expect("the individual append-only growth values are valid");

    assert!(matches!(
        astro_decoder_state_cache_layout.persistence_alignment_token_count(),
        Err(AstroDecoderStateCacheLayoutError::PersistenceAlignmentTokenCountOverflow)
    ));
}

#[test]
fn should_reject_duplicate_tensor_roles_across_one_layer() {
    let rejection =
        AstroDecoderStateCacheLayout::new(vec![AstroDecoderStateCacheLayerLayout::composite(
            vec![
                AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                    AstroDecoderStateCacheTensorLayout::fixed(
                        "state",
                        AstroDecoderStateCacheTensorDtype::Float32,
                        vec![1, 4],
                    ),
                ),
                AstroDecoderStateCacheLayerLayout::recurrent_tensor(
                    AstroDecoderStateCacheTensorLayout::fixed(
                        "state",
                        AstroDecoderStateCacheTensorDtype::Float32,
                        vec![1, 4],
                    ),
                ),
            ],
        )]);

    assert!(matches!(
        rejection,
        Err(AstroDecoderStateCacheLayoutError::DuplicateTensorRole { layer_index: 0, .. })
    ));
}

#[test]
fn should_reject_a_sequence_axis_outside_the_tensor_rank() {
    let rejection = AstroDecoderStateCacheLayout::new(vec![
        AstroDecoderStateCacheLayerLayout::append_only_attention(
            AstroDecoderStateCacheTensorLayout::sequence(
                "attention.keys",
                AstroDecoderStateCacheTensorDtype::BFloat16,
                vec![1, 2, 0, 4],
                4,
            ),
            AstroDecoderStateCacheTensorLayout::sequence(
                "attention.values",
                AstroDecoderStateCacheTensorDtype::BFloat16,
                vec![1, 2, 0, 4],
                2,
            ),
            256,
        ),
    ]);

    assert!(matches!(
        rejection,
        Err(
            AstroDecoderStateCacheLayoutError::SequenceAxisOutsideTensorRank { layer_index: 0, .. }
        )
    ));
}
