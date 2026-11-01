use super::prefill_partial_layer_read::{
    PrefillLayerCompletionRefusals, PrefillStorageReadScope, assembled_prefill_expert_order,
    page_slots_for_expert_ids, prefill_read_expert_ids, prefill_storage_read_scope,
    should_clear_prefill_completion_refusals, should_remember_prefill_completion_refusal,
    split_prefill_route,
};

#[test]
fn should_read_the_entire_unseated_remainder_when_completion_is_still_funded() {
    assert_eq!(
        prefill_storage_read_scope(true, false),
        PrefillStorageReadScope::EntireUnseatedRemainder
    );
}

#[test]
fn should_read_only_routed_misses_after_completion_was_refused() {
    assert_eq!(
        prefill_storage_read_scope(true, true),
        PrefillStorageReadScope::RoutedMissesOnly
    );
}

#[test]
fn should_read_only_routed_misses_when_completion_is_not_funded() {
    assert_eq!(
        prefill_storage_read_scope(false, false),
        PrefillStorageReadScope::RoutedMissesOnly
    );
}

#[test]
fn should_keep_a_routed_miss_read_when_the_remainder_list_is_empty() {
    let missing_route_expert_ids = [4_usize, 9];
    assert_eq!(
        prefill_read_expert_ids(
            PrefillStorageReadScope::EntireUnseatedRemainder,
            &missing_route_expert_ids,
            Some(&[])
        ),
        missing_route_expert_ids
    );
}

#[test]
fn should_read_the_remainder_when_it_is_present() {
    assert_eq!(
        prefill_read_expert_ids(
            PrefillStorageReadScope::EntireUnseatedRemainder,
            &[4, 9],
            Some(&[1, 4, 9, 12])
        ),
        vec![1, 4, 9, 12]
    );
}

#[test]
fn should_read_routed_misses_when_no_remainder_list_exists() {
    assert_eq!(
        prefill_read_expert_ids(
            PrefillStorageReadScope::EntireUnseatedRemainder,
            &[4, 9],
            None
        ),
        vec![4, 9]
    );
}

#[test]
fn should_read_only_the_routed_misses_for_a_miss_scoped_chunk() {
    assert_eq!(
        prefill_read_expert_ids(
            PrefillStorageReadScope::RoutedMissesOnly,
            &[4, 9],
            Some(&[1, 4, 9, 12])
        ),
        vec![4, 9]
    );
}

#[test]
fn should_remember_a_refusal_only_after_a_funded_read_leaves_the_layer_partial() {
    assert!(should_remember_prefill_completion_refusal(
        PrefillStorageReadScope::EntireUnseatedRemainder,
        false
    ));
    assert!(!should_remember_prefill_completion_refusal(
        PrefillStorageReadScope::EntireUnseatedRemainder,
        true
    ));
    assert!(!should_remember_prefill_completion_refusal(
        PrefillStorageReadScope::RoutedMissesOnly,
        false
    ));
}

#[test]
fn should_clear_refusals_only_when_the_ceiling_rises() {
    assert!(should_clear_prefill_completion_refusals(17, 21));
    assert!(!should_clear_prefill_completion_refusals(21, 21));
    assert!(!should_clear_prefill_completion_refusals(21, 17));
}

#[test]
fn should_remember_one_layer_without_marking_its_neighbors() {
    let mut refusals = PrefillLayerCompletionRefusals::default();
    assert!(!refusals.was_refused(39));
    refusals.remember(39);
    assert!(refusals.was_refused(39));
    assert!(!refusals.was_refused(0));
    assert!(!refusals.was_refused(40));
    refusals.clear();
    assert!(!refusals.was_refused(39));
}

#[test]
fn should_split_a_route_into_retained_and_missing_experts() {
    let route_split = split_prefill_route(&[0, 2, 5, 7], &[2, 7, 11]);
    assert_eq!(route_split.retained_expert_ids, vec![0, 5]);
    assert_eq!(route_split.missing_expert_ids, vec![2, 7]);
}

#[test]
fn should_treat_an_empty_route_as_an_empty_split() {
    let route_split = split_prefill_route(&[], &[1, 2]);
    assert!(route_split.retained_expert_ids.is_empty());
    assert!(route_split.missing_expert_ids.is_empty());
}

#[test]
fn should_assign_concatenated_slots_with_retained_experts_first() {
    let expert_order = assembled_prefill_expert_order(&[0, 5], &[2, 7], 8)
        .expect("ids inside the layer must build a slot map");
    assert_eq!(expert_order.expert_ids, vec![0, 5, 2, 7]);
    assert_eq!(
        expert_order.page_slot_by_global_expert_id,
        vec![0, u32::MAX, 2, u32::MAX, u32::MAX, 1, u32::MAX, 3]
    );
}

#[test]
fn should_reject_an_expert_id_outside_the_layer() {
    assert!(assembled_prefill_expert_order(&[0], &[8], 8).is_none());
}

#[test]
fn should_map_streamed_experts_to_their_page_slots() {
    let page_slot_by_global_expert_id = [u32::MAX, 0, u32::MAX, 1];
    assert_eq!(
        page_slots_for_expert_ids(&page_slot_by_global_expert_id, &[3, 1]),
        Some(vec![1, 0])
    );
}

#[test]
fn should_reject_a_streamed_expert_the_page_does_not_hold() {
    let page_slot_by_global_expert_id = [0, u32::MAX];
    assert_eq!(
        page_slots_for_expert_ids(&page_slot_by_global_expert_id, &[1]),
        None
    );
    assert_eq!(
        page_slots_for_expert_ids(&page_slot_by_global_expert_id, &[4]),
        None
    );
}
