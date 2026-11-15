//! Two real replies replayed through the cache.
//!
//! Each fixture records which experts every layer chose for each of 499
//! reply tokens: a summary of a 10,000-token Shakespeare prompt, written by
//! a 48-layer and a 40-layer sparse model. Each replay holds exactly as many
//! experts as the engine held for that model at its memory limit, so the
//! only thing that varies is how the cache spends that memory.

use super::{
    EvictionOrder, ExpertCache, ExpertCacheLayerGeometry, ExpertKey, ExpertTableRules,
    ExpertTables, LayerSlotLimit, RoutedReply,
};

/// Tokens left out of the per-token averages, so the tables a reply starts
/// with empty have filled and the numbers describe steady serving.
const WARM_UP_TOKEN_COUNT: usize = 100;
/// Every expert counts the same here; only the number held matters.
const EXPERT_BYTES: u64 = 1;

struct RecordedModel {
    recording_bytes: &'static [u8],
    held_expert_count: u64,
    /// The layers the engine left partial, with the slots it gave each.
    partial_layer_indexes: [usize; 3],
    partial_layer_slot_count: usize,
}

const FORTY_EIGHT_LAYER_MODEL: RecordedModel = RecordedModel {
    recording_bytes: include_bytes!("fixtures/shakespeare_summary_48_layers.routes"),
    held_expert_count: 11_568,
    partial_layer_indexes: [45, 46, 47],
    partial_layer_slot_count: 16,
};

const FORTY_LAYER_MODEL: RecordedModel = RecordedModel {
    recording_bytes: include_bytes!("fixtures/shakespeare_summary_40_layers.routes"),
    held_expert_count: 9_568,
    partial_layer_indexes: [4, 38, 39],
    partial_layer_slot_count: 32,
};

struct ReplayOutcome {
    cache: ExpertCache,
    reads_per_token_after_warm_up: f64,
}

impl RecordedModel {
    fn routed_reply(&self) -> RoutedReply {
        RoutedReply::from_recording_bytes(self.recording_bytes).expect("a valid fixture")
    }

    /// Replays the reply starting from what the engine held when the reply
    /// began: every layer whole except the three partial ones.
    fn replay(&self, eviction_order: EvictionOrder, divides_by_layer: bool) -> ReplayOutcome {
        let routed_reply = self.routed_reply();
        let experts_per_layer = routed_reply.experts_per_layer();
        let mut slot_counts = vec![usize::from(experts_per_layer); routed_reply.layer_count()];
        for partial_layer_index in self.partial_layer_indexes {
            slot_counts[partial_layer_index] = self.partial_layer_slot_count;
        }
        let layer_slot_limit = if divides_by_layer {
            LayerSlotLimit::FixedPerLayer(slot_counts)
        } else {
            LayerSlotLimit::SharedAcrossLayers
        };
        let layer_geometry = ExpertCacheLayerGeometry {
            expert_count: experts_per_layer,
            expert_payload_bytes: EXPERT_BYTES,
        };
        let mut cache = ExpertCache::new(
            vec![layer_geometry; routed_reply.layer_count()],
            self.held_expert_count * EXPERT_BYTES,
            eviction_order,
            layer_slot_limit,
        );
        self.seat_whole_layers(&mut cache, &routed_reply);
        let reads_per_token_after_warm_up = serve_reply(&mut cache, &routed_reply);
        ReplayOutcome {
            cache,
            reads_per_token_after_warm_up,
        }
    }

    fn seat_whole_layers(&self, cache: &mut ExpertCache, routed_reply: &RoutedReply) {
        for layer_index in 0..routed_reply.layer_count() {
            if self.partial_layer_indexes.contains(&layer_index) {
                continue;
            }
            for expert_id in 0..routed_reply.experts_per_layer() {
                let was_seated = cache.seat(ExpertKey {
                    layer_index,
                    expert_id,
                });
                assert!(was_seated, "the whole layers fit what the engine held");
            }
        }
    }
}

fn serve_reply(cache: &mut ExpertCache, routed_reply: &RoutedReply) -> f64 {
    let mut read_count_after_warm_up = 0_usize;
    for token_index in 0..routed_reply.token_count() {
        cache.begin_token();
        for layer_index in 0..routed_reply.layer_count() {
            let served_route = cache.serve(
                layer_index,
                routed_reply.layer_route(token_index, layer_index),
            );
            if token_index >= WARM_UP_TOKEN_COUNT {
                read_count_after_warm_up +=
                    served_route.read_expert_ids.len() + served_route.unheld_expert_ids.len();
            }
        }
    }
    let measured_token_count = routed_reply.token_count() - WARM_UP_TOKEN_COUNT;
    read_count_after_warm_up as f64 / measured_token_count as f64
}

fn held_never_used_expert_count(cache: &ExpertCache) -> usize {
    let mut held_never_used_expert_count = 0;
    for layer_summary in cache.record().layer_summaries() {
        held_never_used_expert_count += layer_summary.held_never_used_expert_count;
    }
    held_never_used_expert_count
}

#[test]
fn should_reproduce_the_reads_a_division_by_layer_costs() {
    let large_model = FORTY_EIGHT_LAYER_MODEL.replay(EvictionOrder::LeastFrequentlyUsed, true);
    let small_model = FORTY_LAYER_MODEL.replay(EvictionOrder::LeastFrequentlyUsed, true);

    assert!(
        (11.0..14.0).contains(&large_model.reads_per_token_after_warm_up),
        "48-layer reads per token: {}",
        large_model.reads_per_token_after_warm_up
    );
    assert!(
        (9.0..13.0).contains(&small_model.reads_per_token_after_warm_up),
        "40-layer reads per token: {}",
        small_model.reads_per_token_after_warm_up
    );
    assert!(large_model.cache.record().repeat_read_count() > 4_000);
}

#[test]
fn should_hold_experts_no_token_used_when_memory_is_divided_by_layer() {
    let large_model = FORTY_EIGHT_LAYER_MODEL.replay(EvictionOrder::LeastFrequentlyUsed, true);
    let small_model = FORTY_LAYER_MODEL.replay(EvictionOrder::LeastFrequentlyUsed, true);

    assert_eq!(held_never_used_expert_count(&large_model.cache), 2_008);
    assert_eq!(held_never_used_expert_count(&small_model.cache), 2_978);
}

#[test]
fn should_read_under_three_experts_a_token_when_memory_is_shared_across_layers() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let shared = recorded_model.replay(EvictionOrder::LeastRecentlyUsed, false);

        assert!(
            shared.reads_per_token_after_warm_up < 3.0,
            "reads per token: {}",
            shared.reads_per_token_after_warm_up
        );
    }
}

/// The reply needs fewer experts than memory holds, so once the cache has
/// seen what the text uses, the same text must not touch storage at all.
#[test]
fn should_read_nothing_when_the_same_reply_is_served_again() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let routed_reply = recorded_model.routed_reply();
        let mut shared = recorded_model.replay(EvictionOrder::LeastRecentlyUsed, false);
        let read_count_after_first_serving = shared.cache.record().read_count();

        serve_reply(&mut shared.cache, &routed_reply);

        assert_eq!(
            shared.cache.record().read_count(),
            read_count_after_first_serving
        );
    }
}

/// Divided by layer, the same reply keeps reading for as long as it runs.
#[test]
fn should_keep_reading_when_a_divided_cache_serves_the_same_reply_again() {
    let routed_reply = FORTY_EIGHT_LAYER_MODEL.routed_reply();
    let mut divided = FORTY_EIGHT_LAYER_MODEL.replay(EvictionOrder::LeastFrequentlyUsed, true);

    let reads_per_token_on_second_serving = serve_reply(&mut divided.cache, &routed_reply);

    assert!(
        reads_per_token_on_second_serving > 10.0,
        "reads per token: {reads_per_token_on_second_serving}"
    );
}

#[test]
fn should_read_at_least_four_times_less_when_memory_is_shared_across_layers() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let divided = recorded_model.replay(EvictionOrder::LeastFrequentlyUsed, true);
        let shared = recorded_model.replay(EvictionOrder::LeastRecentlyUsed, false);

        assert!(
            shared.reads_per_token_after_warm_up * 4.0 < divided.reads_per_token_after_warm_up,
            "shared {} against divided {}",
            shared.reads_per_token_after_warm_up,
            divided.reads_per_token_after_warm_up
        );
    }
}

#[test]
fn should_read_the_same_under_either_eviction_order_on_these_replies() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let least_recent = recorded_model.replay(EvictionOrder::LeastRecentlyUsed, false);
        let least_frequent = recorded_model.replay(EvictionOrder::LeastFrequentlyUsed, false);

        assert_eq!(
            least_recent.cache.record().read_count(),
            least_frequent.cache.record().read_count()
        );
    }
}

impl RecordedModel {
    /// The engine's tables as the reply begins: every layer whole except the
    /// three partial ones, which start at their small size and empty.
    fn tables_as_the_reply_begins(&self) -> ExpertTables {
        let routed_reply = self.routed_reply();
        let experts_per_layer = routed_reply.experts_per_layer();
        let layer_geometry = ExpertCacheLayerGeometry {
            expert_count: experts_per_layer,
            expert_payload_bytes: EXPERT_BYTES,
        };
        let mut tables = ExpertTables::new(
            vec![layer_geometry; routed_reply.layer_count()],
            self.held_expert_count * EXPERT_BYTES,
            ExpertTableRules::default(),
        );
        let every_expert_id = (0..experts_per_layer).collect::<Vec<_>>();
        for layer_index in 0..routed_reply.layer_count() {
            if self.partial_layer_indexes.contains(&layer_index) {
                tables.adopt_table(layer_index, self.partial_layer_slot_count, &[]);
            } else {
                tables.adopt_table(
                    layer_index,
                    usize::from(experts_per_layer),
                    &every_expert_id,
                );
            }
        }
        tables
    }
}

/// Serves the reply through the tables and returns the reads per token,
/// leaving out the warm-up when `skips_warm_up` is set.
fn serve_reply_from_tables(
    tables: &mut ExpertTables,
    routed_reply: &RoutedReply,
    skips_warm_up: bool,
) -> f64 {
    let first_measured_token = if skips_warm_up {
        WARM_UP_TOKEN_COUNT
    } else {
        0
    };
    let read_count_before = tables.record().read_count();
    let mut read_count_during_warm_up = 0;
    for token_index in 0..routed_reply.token_count() {
        if token_index == first_measured_token {
            read_count_during_warm_up = tables.record().read_count() - read_count_before;
        }
        tables.begin_token();
        for layer_index in 0..routed_reply.layer_count() {
            tables.serve(
                layer_index,
                routed_reply.layer_route(token_index, layer_index),
            );
        }
    }
    let measured_read_count =
        tables.record().read_count() - read_count_before - read_count_during_warm_up;
    let measured_token_count = routed_reply.token_count() - first_measured_token;
    measured_read_count as f64 / measured_token_count as f64
}

#[test]
fn should_read_under_two_experts_a_token_from_tables_that_grow_and_compact() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let routed_reply = recorded_model.routed_reply();
        let mut tables = recorded_model.tables_as_the_reply_begins();

        let reads_per_token = serve_reply_from_tables(&mut tables, &routed_reply, true);

        assert!(reads_per_token < 2.0, "reads per token: {reads_per_token}");
    }
}

#[test]
fn should_read_almost_nothing_from_tables_when_the_same_reply_is_served_again() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let routed_reply = recorded_model.routed_reply();
        let mut tables = recorded_model.tables_as_the_reply_begins();
        serve_reply_from_tables(&mut tables, &routed_reply, false);

        let reads_per_token = serve_reply_from_tables(&mut tables, &routed_reply, false);

        assert!(reads_per_token < 0.3, "reads per token: {reads_per_token}");
    }
}

/// Reallocating a table copies what it holds, so the design only works if
/// reallocations stay rare and most layers stay whole.
#[test]
fn should_reallocate_rarely_and_leave_most_layers_whole() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let routed_reply = recorded_model.routed_reply();
        let mut tables = recorded_model.tables_as_the_reply_begins();

        serve_reply_from_tables(&mut tables, &routed_reply, false);

        let mut partial_layer_count = 0;
        for layer_index in 0..routed_reply.layer_count() {
            if tables.held_expert_count(layer_index) < usize::from(routed_reply.experts_per_layer())
            {
                partial_layer_count += 1;
            }
        }
        assert!(
            tables.reallocation_count() < 120,
            "reallocations: {}",
            tables.reallocation_count()
        );
        assert!(
            partial_layer_count <= 12,
            "partial layers: {partial_layer_count}"
        );
    }
}

#[test]
fn should_never_take_more_memory_than_the_capacity() {
    for recorded_model in [&FORTY_EIGHT_LAYER_MODEL, &FORTY_LAYER_MODEL] {
        let routed_reply = recorded_model.routed_reply();
        let mut tables = recorded_model.tables_as_the_reply_begins();
        let capacity_bytes = recorded_model.held_expert_count * EXPERT_BYTES;

        for token_index in 0..routed_reply.token_count() {
            tables.begin_token();
            for layer_index in 0..routed_reply.layer_count() {
                tables.serve(
                    layer_index,
                    routed_reply.layer_route(token_index, layer_index),
                );
                assert!(tables.allocated_bytes() <= capacity_bytes);
            }
        }
    }
}
