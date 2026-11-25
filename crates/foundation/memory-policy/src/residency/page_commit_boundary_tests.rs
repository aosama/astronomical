//! Boundary, positive, and negative tests for the multi-token union
//! complete-layer adoption gate.

use crate::residency::ExpertLayerState;
use crate::residency::{
    should_commit_multi_token_union_as_complete_layer, should_seed_partial_layer_on_prefill_read,
};

#[test]
fn should_commit_a_multi_token_union_as_a_complete_layer_when_the_plan_funds_it() {
    // The 2026-10-23 Ornith under-utilization flight: a 2,048-token chunk
    // routes every expert of the layer, paying the complete-layer read cost,
    // and the plan funds the claim, so the union must be offered to retained
    // ownership instead of re-read from storage on every later chunk.
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
}

#[test]
fn should_never_commit_a_multi_token_union_that_is_not_the_whole_layer() {
    // A partial union is not a complete layer: adopting it as one would
    // claim the whole layer's payload budget for a fraction of the experts.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        96,
        128,
    ));
    // One-off and empty unions never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        0,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        true,
        ExpertLayerState::Kept,
        128,
        0,
    ));
}

#[test]
fn should_not_commit_a_multi_token_union_for_unfunded_or_non_production_plans() {
    // Only a layer that should be kept whole authorizes the adoption offer.
    for unfunded_target in [ExpertLayerState::Absent, ExpertLayerState::Hot] {
        assert!(
            !should_commit_multi_token_union_as_complete_layer(
                2_048,
                true,
                unfunded_target,
                128,
                128,
            ),
            "target {unfunded_target:?} must not authorize the offer"
        );
    }
    // Diagnostic paging modes and single-token decode forwards never adopt.
    assert!(!should_commit_multi_token_union_as_complete_layer(
        2_048,
        false,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        1,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
    assert!(!should_commit_multi_token_union_as_complete_layer(
        0,
        true,
        ExpertLayerState::Kept,
        128,
        128,
    ));
}

#[test]
fn should_seed_a_partial_prefill_union_for_a_funded_complete_layer_target() {
    let funded_target = ExpertLayerState::Kept;
    for routed_expert_count in [1, 127] {
        assert!(
            should_seed_partial_layer_on_prefill_read(
                2_048,
                true,
                funded_target,
                0,
                routed_expert_count,
                128,
            ),
            "a cold non-empty partial union must seed for {funded_target:?} \
             at {routed_expert_count} routed experts"
        );
    }
}

#[test]
fn should_not_seed_a_partial_prefill_union_outside_its_funded_boundary() {
    let funded_target = ExpertLayerState::Kept;
    for routed_expert_count in [0, 128, 129] {
        assert!(!should_seed_partial_layer_on_prefill_read(
            2_048,
            true,
            funded_target,
            0,
            routed_expert_count,
            128,
        ));
    }
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        true,
        funded_target,
        0,
        0,
        0,
    ));
    // An existing partial table uses the missing-expert growth path instead.
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        true,
        funded_target,
        1,
        64,
        128,
    ));
    // Decode tokens, diagnostic paths, and absent or partial-only plans do not
    // authorize a prefill completion seed.
    assert!(!should_seed_partial_layer_on_prefill_read(
        1,
        true,
        funded_target,
        0,
        64,
        128,
    ));
    assert!(!should_seed_partial_layer_on_prefill_read(
        0,
        true,
        funded_target,
        0,
        64,
        128,
    ));
    assert!(!should_seed_partial_layer_on_prefill_read(
        2_048,
        false,
        funded_target,
        0,
        64,
        128,
    ));
    for unfunded_target in [ExpertLayerState::Absent, ExpertLayerState::Hot] {
        assert!(
            !should_seed_partial_layer_on_prefill_read(2_048, true, unfunded_target, 0, 64, 128,),
            "target {unfunded_target:?} must not authorize the seed"
        );
    }
}

#[test]
fn should_complete_a_partial_layer_when_the_plan_funds_the_remaining_experts() {
    use crate::residency::ExpertLayerState;
    use crate::residency::should_complete_partial_layer_on_prefill_read;
    assert!(
        should_complete_partial_layer_on_prefill_read(true, ExpertLayerState::Kept, 65, 128),
        "a partly held layer that should be kept whole converges on the next read"
    );
}

#[test]
fn should_not_complete_an_empty_complete_or_unfunded_layer() {
    use crate::residency::ExpertLayerState;
    use crate::residency::should_complete_partial_layer_on_prefill_read;
    let funded = ExpertLayerState::Kept;
    // Cold-union seeding and complete-union adoption own the empty-table case.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 0, 128
    ));
    // A complete table has nothing left to read.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 128, 128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 200, 128
    ));
    // Unfunded or absent plans, diagnostic modes, and degenerate capacities
    // never authorize the extended read.
    assert!(!should_complete_partial_layer_on_prefill_read(
        true,
        ExpertLayerState::Absent,
        65,
        128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        false, funded, 65, 128
    ));
    assert!(!should_complete_partial_layer_on_prefill_read(
        true, funded, 65, 0
    ));
}

mod decode_whole_layer_keep {
    use crate::residency::{
        CurrentExpertLayerResidency, DECODE_TABLE_TARGET_SLOT_COUNT, RetainedExpertPageClass,
        complete_layers_beyond_keep_count, decode_table_target_slot_count,
        decode_whole_layer_keep_count,
    };

    const EXPERT_BYTES: u64 = 1_000;

    fn capacity_of(expert_count: u64) -> u64 {
        expert_count * EXPERT_BYTES
    }

    #[test]
    fn should_keep_five_whole_layers_at_half_the_model_s_size() {
        // The anchor: 4,672 experts of room, 40 layers of 256. Five whole
        // layers leave 3,392 for 35 tables, 96 each; a sixth would leave 92.
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(4_672), EXPERT_BYTES, 256, 40),
            5
        );
    }

    #[test]
    fn should_keep_more_whole_layers_as_the_capacity_grows() {
        // Three quarters of the model: 7,350 experts of room.
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(7_350), EXPERT_BYTES, 256, 40),
            21
        );
    }

    #[test]
    fn should_keep_every_layer_whole_when_the_whole_model_fits() {
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(256 * 40), EXPERT_BYTES, 256, 40),
            40
        );
    }

    #[test]
    fn should_keep_no_layer_whole_when_the_tables_need_all_the_room() {
        let just_below_one_whole_layer = (256 + DECODE_TABLE_TARGET_SLOT_COUNT * 39) as u64 - 1;
        assert_eq!(
            decode_whole_layer_keep_count(
                capacity_of(just_below_one_whole_layer),
                EXPERT_BYTES,
                256,
                40
            ),
            0
        );
        assert_eq!(decode_whole_layer_keep_count(0, EXPERT_BYTES, 256, 40), 0);
    }

    #[test]
    fn should_keep_no_layer_whole_for_a_layer_without_size() {
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(5_000), 0, 256, 40),
            0
        );
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(5_000), EXPERT_BYTES, 0, 40),
            0
        );
    }

    #[test]
    fn should_cap_the_table_target_at_a_small_layer_s_size() {
        // Layers of 64 experts: a table's target is the whole layer, so the
        // tables themselves fill whole and no layer needs keeping whole
        // first; with one expert short of the model, none is.
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(64 * 10 - 1), EXPERT_BYTES, 64, 10),
            0
        );
    }

    #[test]
    fn should_protect_three_eighths_of_a_layer_larger_than_256_experts() {
        assert_eq!(decode_table_target_slot_count(512), 192);
        assert_eq!(decode_table_target_slot_count(1_024), 384);
    }

    #[test]
    fn should_protect_the_measured_96_experts_for_layers_of_up_to_256() {
        assert_eq!(decode_table_target_slot_count(256), 96);
        assert_eq!(decode_table_target_slot_count(128), 96);
        // One expert past 256 still rounds down to the measured target.
        assert_eq!(decode_table_target_slot_count(257), 96);
        assert_eq!(decode_table_target_slot_count(64), 64, "at most the layer");
        assert_eq!(decode_table_target_slot_count(0), 0);
    }

    #[test]
    fn should_keep_no_layer_whole_for_512_expert_layers_at_a_third_of_the_model() {
        // Qwen3.8-Flash-Next at a 34 GB limit: about 9,140 experts of room,
        // 48 layers of 512. One whole layer would leave 47 tables 183 each,
        // short of 192; the fixed 96 kept ten layers whole here.
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(9_140), EXPERT_BYTES, 512, 48),
            0
        );
        // A little more room keeps one.
        assert_eq!(
            decode_whole_layer_keep_count(capacity_of(9_800), EXPERT_BYTES, 512, 48),
            1
        );
    }

    fn residency(
        layer_index: usize,
        class: RetainedExpertPageClass,
        covered_weighted_demand: u64,
    ) -> CurrentExpertLayerResidency {
        CurrentExpertLayerResidency {
            layer_index,
            class,
            retained_expert_ids: Vec::new(),
            payload_bytes: 0,
            covered_weighted_demand,
        }
    }

    #[test]
    fn should_give_up_the_least_demanded_complete_layers_beyond_the_keep_count() {
        let layer_residencies = [
            residency(0, RetainedExpertPageClass::StableCompleteLayer, 50),
            residency(1, RetainedExpertPageClass::StableCompleteLayer, 900),
            residency(2, RetainedExpertPageClass::ElasticRoutedExperts, 1),
            residency(3, RetainedExpertPageClass::StableCompleteLayer, 400),
        ];

        assert_eq!(
            complete_layers_beyond_keep_count(&layer_residencies, 1),
            vec![3, 0]
        );
    }

    #[test]
    fn should_give_up_nothing_when_the_keep_count_covers_every_complete_layer() {
        let layer_residencies = [
            residency(0, RetainedExpertPageClass::StableCompleteLayer, 5),
            residency(1, RetainedExpertPageClass::ElasticRoutedExperts, 5),
        ];

        assert!(complete_layers_beyond_keep_count(&layer_residencies, 1).is_empty());
        assert!(complete_layers_beyond_keep_count(&[], 0).is_empty());
    }

    #[test]
    fn should_break_a_demand_tie_by_layer_order() {
        let layer_residencies = [
            residency(4, RetainedExpertPageClass::StableCompleteLayer, 7),
            residency(2, RetainedExpertPageClass::StableCompleteLayer, 7),
        ];

        assert_eq!(
            complete_layers_beyond_keep_count(&layer_residencies, 1),
            vec![4]
        );
    }
}
