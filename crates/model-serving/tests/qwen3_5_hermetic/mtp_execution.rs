use astronomical_model_serving::{
    MtpDepthDowngradeReason, MtpDraftDepth, MtpMemoryCandidate, MtpMemoryProjection,
    Qwen3_5MtpRequestEligibility, Qwen3_5MtpRequestEligibilityInputs,
    predictor_history_requires_verified_hidden_replay,
    qwen3_5_mtp_effective_depth_and_reason_for_windows, qwen3_5_mtp_effective_depth_for_windows,
    qwen3_5_mtp_memory_admission, qwen3_5_mtp_request_eligibility,
    qwen3_5_mtp_verification_decision, qwen3_5_mtp_verification_transient_array_bytes,
};

#[test]
fn should_explain_each_user_visible_target_only_request_class() {
    let eligible_inputs = Qwen3_5MtpRequestEligibilityInputs {
        mtp_enabled: true,
        mtp_runtime_is_active: true,
        model_has_mtp_weights: true,
        sampling_is_greedy: true,
        has_precomputed_visual_embeddings: false,
        has_processed_visual_images: false,
        persistent_prompt_cache_is_available: false,
        prompt_token_count: 3,
        restored_prompt_token_count: 0,
    };
    assert_eq!(
        qwen3_5_mtp_request_eligibility(eligible_inputs),
        Qwen3_5MtpRequestEligibility::Eligible
    );

    let ineligible_cases = [
        (
            Qwen3_5MtpRequestEligibilityInputs {
                sampling_is_greedy: false,
                ..eligible_inputs
            },
            Qwen3_5MtpRequestEligibility::SampledDecoding,
        ),
        (
            Qwen3_5MtpRequestEligibilityInputs {
                has_processed_visual_images: true,
                ..eligible_inputs
            },
            Qwen3_5MtpRequestEligibility::ProcessedVisionInput,
        ),
        (
            Qwen3_5MtpRequestEligibilityInputs {
                persistent_prompt_cache_is_available: true,
                ..eligible_inputs
            },
            Qwen3_5MtpRequestEligibility::PersistentPromptCacheAvailable,
        ),
        (
            Qwen3_5MtpRequestEligibilityInputs {
                restored_prompt_token_count: 2,
                ..eligible_inputs
            },
            Qwen3_5MtpRequestEligibility::InsufficientUncachedPromptHistory,
        ),
    ];
    for (inputs, expected_reason) in ineligible_cases {
        let eligibility = qwen3_5_mtp_request_eligibility(inputs);
        assert_eq!(eligibility, expected_reason);
        assert!(!eligibility.identifier().is_empty());
    }
}

#[test]
fn should_replay_only_accepted_drafts_with_verified_hidden_rows() {
    assert!(!predictor_history_requires_verified_hidden_replay(0));
    assert!(predictor_history_requires_verified_hidden_replay(1));
    assert!(predictor_history_requires_verified_hidden_replay(2));
    assert!(predictor_history_requires_verified_hidden_replay(3));
}

#[test]
fn should_accept_every_longest_prefix_at_depths_one_through_three() {
    for requested_depth in 1_u8..=3 {
        let draft_token_ids = (0..requested_depth)
            .map(|draft_index| 100 + u32::from(draft_index))
            .collect::<Vec<_>>();
        for expected_accepted_count in 0_u8..=requested_depth {
            let mut target_token_ids = Vec::with_capacity(usize::from(requested_depth) + 1);
            for draft_index in 0..requested_depth {
                let target_token_id = if draft_index < expected_accepted_count {
                    draft_token_ids[usize::from(draft_index)]
                } else {
                    900 + u32::from(draft_index)
                };
                target_token_ids.push(target_token_id);
            }
            target_token_ids.push(1_000 + u32::from(requested_depth));

            let decision = qwen3_5_mtp_verification_decision(
                MtpDraftDepth::new(requested_depth).expect("test depth should be valid"),
                &draft_token_ids,
                &target_token_ids,
                &[],
            )
            .expect("matching verification vectors should be accepted");

            assert_eq!(decision.proposed_count(), requested_depth);
            assert_eq!(decision.accepted_count(), expected_accepted_count);
            assert_eq!(
                decision.pending_target_token_id(),
                Some(target_token_ids[usize::from(expected_accepted_count)])
            );
            assert!(!decision.was_eos_truncated());
        }
    }
}

#[test]
fn should_truncate_at_the_first_eos_in_an_accepted_depth_three_prefix() {
    let draft_token_ids = [101, 102, 103];
    for eos_draft_position in 0_usize..3 {
        let mut eos_specific_drafts = draft_token_ids;
        eos_specific_drafts[eos_draft_position] = 2;
        let target_token_ids = [
            eos_specific_drafts[0],
            eos_specific_drafts[1],
            eos_specific_drafts[2],
            700,
        ];
        let decision = qwen3_5_mtp_verification_decision(
            MtpDraftDepth::new(3).expect("depth three should be valid"),
            &eos_specific_drafts,
            &target_token_ids,
            &[2],
        )
        .expect("EOS decision should be valid");

        assert_eq!(decision.accepted_count(), (eos_draft_position + 1) as u8);
        assert_eq!(decision.pending_target_token_id(), None);
        assert!(decision.was_eos_truncated());
    }
}

#[test]
fn should_preserve_eos_when_it_is_the_correction_or_bonus() {
    for (draft_token_ids, target_token_ids) in [
        (vec![101, 102], vec![2, 702, 703]),
        (vec![101, 102], vec![101, 102, 2]),
    ] {
        let decision = qwen3_5_mtp_verification_decision(
            MtpDraftDepth::new(2).expect("depth two should be valid"),
            &draft_token_ids,
            &target_token_ids,
            &[2],
        )
        .expect("correction or bonus decision should be valid");

        assert_eq!(decision.pending_target_token_id(), Some(2));
        assert!(!decision.was_eos_truncated());
    }
}

#[test]
fn should_clamp_depth_to_output_context_and_thinking_windows() {
    let requested_depth = MtpDraftDepth::new(3).expect("depth three should be valid");
    let cases = [
        (7, 12, 20, 40, false, 0, None, Some(3)),
        (8, 12, 20, 40, false, 0, None, Some(3)),
        (7, 12, 37, 40, false, 0, None, Some(2)),
        (7, 12, 20, 40, true, 6, Some(10), Some(2)),
        (11, 12, 20, 40, false, 0, None, None),
    ];
    for (
        generated_token_count,
        maximum_output_tokens,
        next_position_tokens,
        maximum_position_count,
        is_inside_thinking,
        thinking_token_count,
        thinking_budget,
        expected_depth,
    ) in cases
    {
        let effective_depth = qwen3_5_mtp_effective_depth_for_windows(
            requested_depth,
            generated_token_count,
            maximum_output_tokens,
            next_position_tokens,
            maximum_position_count,
            is_inside_thinking,
            thinking_token_count,
            thinking_budget,
        );
        assert_eq!(effective_depth.map(MtpDraftDepth::get), expected_depth);
    }
}

#[test]
fn should_attribute_each_non_memory_depth_downgrade_to_its_limiting_window() {
    let requested_depth = MtpDraftDepth::new(3).expect("depth three should be valid");
    let cases = [
        (
            (10, 12, 20, 40, false, 0, None),
            MtpDepthDowngradeReason::OutputWindow,
        ),
        (
            (0, 12, 38, 40, false, 0, None),
            MtpDepthDowngradeReason::ContextWindow,
        ),
        (
            (0, 12, 20, 40, true, 7, Some(10)),
            MtpDepthDowngradeReason::ThinkingWindow,
        ),
    ];
    for (
        (
            generated_token_count,
            maximum_output_tokens,
            next_position_tokens,
            maximum_position_count,
            is_inside_thinking,
            thinking_token_count,
            thinking_budget,
        ),
        expected_reason,
    ) in cases
    {
        let (effective_depth, downgrade_reason) =
            qwen3_5_mtp_effective_depth_and_reason_for_windows(
                requested_depth,
                generated_token_count,
                maximum_output_tokens,
                next_position_tokens,
                maximum_position_count,
                is_inside_thinking,
                thinking_token_count,
                thinking_budget,
            );

        assert_eq!(effective_depth.map(MtpDraftDepth::get), Some(1));
        assert_eq!(downgrade_reason, Some(expected_reason));
    }
}

#[test]
fn should_fall_back_from_depth_three_to_target_only_at_exact_memory_boundaries() {
    let candidates = [
        MtpMemoryCandidate::new(MtpDraftDepth::new(3).expect("valid depth"), 300),
        MtpMemoryCandidate::new(MtpDraftDepth::new(2).expect("valid depth"), 200),
        MtpMemoryCandidate::new(MtpDraftDepth::DEPTH_ONE, 100),
    ];
    for (available_bytes, expected_depth) in
        [(300, Some(3)), (299, Some(2)), (199, Some(1)), (99, None)]
    {
        let admission = qwen3_5_mtp_memory_admission(&candidates, available_bytes);
        assert_eq!(
            admission.effective_depth().map(MtpDraftDepth::get),
            expected_depth
        );
        assert_eq!(
            admission.downgrade_reason(),
            if expected_depth == Some(3) {
                None
            } else {
                Some(MtpDepthDowngradeReason::Memory)
            }
        );
    }
}

#[test]
fn should_project_actual_sequential_mtp_growth_instead_of_depth_one_overprojection() {
    let projection = MtpMemoryProjection::new(MtpDraftDepth::DEPTH_ONE, 64, 96, 128, 32, 16, 8)
        .expect("bounded projection should not overflow");

    assert_eq!(projection.mtp_persistent_growth_bytes(), 64);
    assert_eq!(projection.target_persistent_growth_bytes(), 96);
    assert_eq!(projection.total_required_bytes(), 344);
}

#[test]
fn should_reject_verification_vectors_that_do_not_match_the_effective_depth() {
    let depth = MtpDraftDepth::new(2).expect("depth two should be valid");
    assert!(qwen3_5_mtp_verification_decision(depth, &[101], &[101, 102, 103], &[]).is_err());
    assert!(qwen3_5_mtp_verification_decision(depth, &[101, 102], &[101, 102], &[]).is_err());
}

#[test]
fn should_keep_verification_transient_arrays_independent_of_boundary_snapshots() {
    let depth_one = qwen3_5_mtp_verification_transient_array_bytes(MtpDraftDepth::DEPTH_ONE, 8, 4)
        .expect("depth-one transient arrays should not overflow");
    let depth_two = qwen3_5_mtp_verification_transient_array_bytes(
        MtpDraftDepth::new(2).expect("depth two should be valid"),
        8,
        4,
    )
    .expect("depth-two transient arrays should not overflow");
    let projection = MtpMemoryProjection::new(
        MtpDraftDepth::new(2).expect("depth two should be valid"),
        64,
        96,
        128,
        32,
        depth_two,
        8,
    )
    .expect("bounded projection should not overflow");

    // Depth two has one extra draft plus one extra verifier row versus depth one.
    assert!(depth_two > depth_one);
    assert_eq!(
        projection.total_required_bytes(),
        64 + 96 + 128 + 32 + depth_two + 8
    );
}
