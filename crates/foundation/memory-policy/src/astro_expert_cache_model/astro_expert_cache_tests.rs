use super::{
    AstroExpertCacheLayerGeometry, DivisibleAstroExpertCache, EvictionOrder, ExpertKey,
    LayerSlotLimit,
};

const EXPERT_BYTES: u64 = 10;

fn two_layer_astro_expert_cache(
    capacity_expert_count: u64,
    eviction_order: EvictionOrder,
) -> DivisibleAstroExpertCache {
    DivisibleAstroExpertCache::new(
        vec![
            AstroExpertCacheLayerGeometry {
                expert_count: 8,
                expert_payload_bytes: EXPERT_BYTES,
            };
            2
        ],
        capacity_expert_count * EXPERT_BYTES,
        eviction_order,
        LayerSlotLimit::SharedAcrossLayers,
    )
}

fn expert(layer_index: usize, expert_id: u16) -> ExpertKey {
    ExpertKey {
        layer_index,
        expert_id,
    }
}

#[test]
fn should_read_a_missing_expert_once_and_hit_it_afterwards() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(4, EvictionOrder::LeastRecentlyUsed);

    astro_expert_cache.begin_token();
    let first_serving = astro_expert_cache.serve(0, &[1, 2]);
    astro_expert_cache.begin_token();
    let second_serving = astro_expert_cache.serve(0, &[1, 2]);

    assert_eq!(first_serving.read_expert_ids, vec![1, 2]);
    assert!(second_serving.read_expert_ids.is_empty());
    let layer_summary = astro_expert_cache.record().layer_summaries()[0];
    assert_eq!(layer_summary.lookup_count, 4);
    assert_eq!(layer_summary.hit_count, 2);
    assert_eq!(layer_summary.first_read_count, 2);
    assert_eq!(layer_summary.repeat_read_count, 0);
}

#[test]
fn should_give_up_an_idle_expert_in_another_layer_for_a_busy_one() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(4, EvictionOrder::LeastRecentlyUsed);
    for expert_id in 0..4 {
        assert!(astro_expert_cache.seat(expert(0, expert_id)));
    }

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(1, &[5]);

    assert_eq!(serving.read_expert_ids, vec![5]);
    assert_eq!(serving.evicted, vec![expert(0, 0)]);
    assert_eq!(astro_expert_cache.held_expert_count(0), 3);
    assert_eq!(astro_expert_cache.held_expert_count(1), 1);
    let eviction = astro_expert_cache
        .record()
        .eviction_events()
        .next()
        .copied()
        .expect("one eviction");
    assert_eq!(eviction.needed, expert(1, 5));
    assert_eq!(eviction.evicted_use_count, 0);
}

#[test]
fn should_give_up_the_least_recently_used_expert_first() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(2, EvictionOrder::LeastRecentlyUsed);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[2]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(1, &[7]);

    assert_eq!(serving.evicted, vec![expert(0, 2)]);
}

#[test]
fn should_give_up_the_least_frequently_used_expert_first() {
    let mut astro_expert_cache =
        two_layer_astro_expert_cache(2, EvictionOrder::LeastFrequentlyUsed);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[2]);

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(1, &[7]);

    assert_eq!(serving.evicted, vec![expert(0, 2)]);
}

#[test]
fn should_count_a_read_of_an_expert_it_gave_up_as_a_repeat_read() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(1, EvictionOrder::LeastRecentlyUsed);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[2]);

    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);

    assert_eq!(astro_expert_cache.record().read_count(), 3);
    assert_eq!(astro_expert_cache.record().repeat_read_count(), 1);
    assert_eq!(
        astro_expert_cache.record().expert_uses(0)[1].eviction_count,
        1
    );
}

#[test]
fn should_serve_without_keeping_when_one_token_needs_more_than_fits() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(2, EvictionOrder::LeastRecentlyUsed);

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(0, &[1, 2, 3]);

    assert_eq!(serving.read_expert_ids, vec![1, 2]);
    assert_eq!(serving.unheld_expert_ids, vec![3]);
    assert!(serving.evicted.is_empty());
    assert_eq!(
        astro_expert_cache.record().layer_summaries()[0].unheld_read_count,
        1
    );
}

#[test]
fn should_hold_nothing_when_capacity_is_zero() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(0, EvictionOrder::LeastRecentlyUsed);

    assert!(!astro_expert_cache.seat(expert(0, 0)));
    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(0, &[1]);

    assert_eq!(serving.unheld_expert_ids, vec![1]);
    assert_eq!(astro_expert_cache.held_bytes(), 0);
}

#[test]
fn should_read_an_expert_named_twice_in_one_route_once() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(4, EvictionOrder::LeastRecentlyUsed);

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(0, &[1, 1]);

    assert_eq!(serving.read_expert_ids, vec![1]);
    assert_eq!(astro_expert_cache.record().read_count(), 1);
}

#[test]
fn should_keep_a_full_layer_within_its_fixed_slots() {
    let mut astro_expert_cache = DivisibleAstroExpertCache::new(
        vec![
            AstroExpertCacheLayerGeometry {
                expert_count: 8,
                expert_payload_bytes: EXPERT_BYTES,
            };
            2
        ],
        16 * EXPERT_BYTES,
        EvictionOrder::LeastRecentlyUsed,
        LayerSlotLimit::FixedPerLayer(vec![8, 1]),
    );
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(1, &[3]);

    astro_expert_cache.begin_token();
    let serving = astro_expert_cache.serve(1, &[4]);

    assert_eq!(serving.evicted, vec![expert(1, 3)]);
    assert_eq!(astro_expert_cache.held_expert_count(1), 1);
}

#[test]
fn should_give_back_the_least_valuable_experts_when_capacity_shrinks() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(3, EvictionOrder::LeastRecentlyUsed);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[2]);
    astro_expert_cache.begin_token();
    astro_expert_cache.serve(1, &[3]);

    let evicted = astro_expert_cache.set_capacity(EXPERT_BYTES);

    assert_eq!(evicted, vec![expert(0, 1), expert(0, 2)]);
    assert_eq!(astro_expert_cache.held_bytes(), EXPERT_BYTES);
    assert!(astro_expert_cache.is_held(expert(1, 3)));
}

#[test]
fn should_report_held_experts_that_no_token_used() {
    let mut astro_expert_cache = two_layer_astro_expert_cache(4, EvictionOrder::LeastRecentlyUsed);
    assert!(astro_expert_cache.seat(expert(0, 0)));
    assert!(astro_expert_cache.seat(expert(0, 1)));

    astro_expert_cache.begin_token();
    astro_expert_cache.serve(0, &[1]);

    let layer_summary = astro_expert_cache.record().layer_summaries()[0];
    assert_eq!(layer_summary.held_expert_count, 2);
    assert_eq!(layer_summary.used_expert_count, 1);
    assert_eq!(layer_summary.held_never_used_expert_count, 1);
}
