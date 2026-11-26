//! Direct-MLX contracts of a route looked up through a layer's table: every
//! assignment gets the slot of its expert, a missing expert is read into the
//! table first, and a route that would need a larger table is refused with
//! the table as it was.

use serial_test::file_serial;

use super::direct_fetch_tests::{
    cache_holding_experts_zero_and_one, expected_words_for, words_held_for,
};
use super::*;
use crate::qwen3_5_text::expert_streaming::ExpertSlotStorageFiller;
use crate::qwen3_5_text::expert_streaming::expert_slot_storage_fill_tests::{
    FileLayout, layer_plan, test_runtime, write_expert_file,
};
use engine_core::expert_paging::QuantizedExpertLayerPlan;
use runtime_integration::MlxRuntime;

/// A four-expert file on disk, its layer plan, and a filler that reads it.
struct ExpertStorage {
    runtime: MlxRuntime,
    plan: QuantizedExpertLayerPlan,
    filler: ExpertSlotStorageFiller,
    /// Kept so the file outlives the reads.
    expert_file: tempfile::NamedTempFile,
}

fn expert_storage() -> ExpertStorage {
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    ExpertStorage {
        runtime: test_runtime(),
        filler: ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
            .expect("the filler opens"),
        plan,
        expert_file,
    }
}

fn assignments(expert_ids: &[u32]) -> Vec<[u8; 4]> {
    let mut expert_assignments = Vec::with_capacity(expert_ids.len());
    for expert_id in expert_ids {
        expert_assignments.push(expert_id.to_ne_bytes());
    }
    expert_assignments
}

fn slots_written(table_slots: &[[u8; 4]]) -> Vec<u32> {
    let mut slots = Vec::with_capacity(table_slots.len());
    for table_slot in table_slots {
        slots.push(u32::from_ne_bytes(*table_slot));
    }
    slots
}

#[test]
#[file_serial]
fn should_give_every_assignment_its_experts_slot_without_reading_when_all_are_held() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let mut table_slots = vec![[0xFF_u8; 4]; 3];

    let read_through = astro_expert_cache
        .read_route_through_table(
            &runtime,
            0,
            &plan,
            &assignments(&[1, 0, 1]),
            2,
            &filler,
            &mut table_slots,
        )
        .expect("the lookup runs")
        .expect("the table serves the route");

    let slots = slots_written(&table_slots);
    assert_eq!(slots[0], slots[2], "one expert, one slot");
    assert_ne!(slots[0], slots[1]);
    assert_eq!(read_through.retained_expert_count, 2);
    assert_eq!(read_through.missing_expert_count, 0);
    assert!(read_through.table_fetch_report.is_none());
}

#[test]
#[file_serial]
fn should_read_a_missing_expert_into_a_full_table_and_name_its_slot() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    // Two slots, both taken, no room to grow: expert 1 gives way to 3.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let mut table_slots = vec![[0xFF_u8; 4]; 3];

    let read_through = astro_expert_cache
        .read_route_through_table(
            &runtime,
            0,
            &plan,
            &assignments(&[0, 3, 0]),
            2,
            &filler,
            &mut table_slots,
        )
        .expect("the lookup runs")
        .expect("the table serves the route");

    assert_eq!(read_through.retained_expert_count, 1);
    assert_eq!(read_through.missing_expert_count, 1);
    assert_eq!(
        read_through
            .table_fetch_report
            .expect("one expert was read")
            .storage_read_count,
        1
    );
    let slots = slots_written(&table_slots);
    assert_eq!(slots[0], slots[2]);
    assert_ne!(slots[0], slots[1]);
    assert!(
        slots.iter().all(|slot| *slot < 2),
        "slots of a two-slot table"
    );
    for expert_id in [0, 3] {
        assert_eq!(
            words_held_for(&astro_expert_cache, expert_id),
            expected_words_for(expert_id),
            "expert {expert_id} is in the table with its own bytes"
        );
    }
}

#[test]
#[file_serial]
fn should_refuse_a_route_whose_fetch_would_grow_the_table() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    // Two slots taken, and the reply's plan allows four: a fetch from the
    // model's thread would grow the table for expert 3.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    let mut table_slots = vec![[0xFF_u8; 4]; 2];

    let read_through = astro_expert_cache
        .read_route_through_table(
            &runtime,
            0,
            &plan,
            &assignments(&[0, 3]),
            4,
            &filler,
            &mut table_slots,
        )
        .expect("the lookup runs");

    assert_eq!(read_through.err(), Some(TableRefusal::WouldGrow));
    let coverage = astro_expert_cache.routed_expert_coverage(0, &[0, 1, 3]);
    assert_eq!(
        coverage.retained_expert_ids,
        vec![0, 1],
        "the table is as it was"
    );
    assert_eq!(coverage.missing_expert_ids, vec![3]);
}

#[test]
#[file_serial]
fn should_refuse_a_layer_with_no_table_and_a_route_that_needs_every_slot_and_one_more() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    let mut empty_cache = AstroExpertCache::new(1);
    empty_cache.set_capacity(1024 * 1024 * 1024);
    let mut one_slot = vec![[0xFF_u8; 4]; 1];
    assert!(
        empty_cache
            .read_route_through_table(
                &runtime,
                0,
                &plan,
                &assignments(&[3]),
                2,
                &filler,
                &mut one_slot
            )
            .expect("the lookup runs")
            .err()
            == Some(TableRefusal::TableUnavailable),
        "a layer with no table is served another way"
    );

    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let mut three_slots = vec![[0xFF_u8; 4]; 3];
    assert!(
        astro_expert_cache
            .read_route_through_table(
                &runtime,
                0,
                &plan,
                &assignments(&[0, 1, 3]),
                2,
                &filler,
                &mut three_slots,
            )
            .expect("the lookup runs")
            .err()
            == Some(TableRefusal::CannotSeatEveryExpert),
        "no expert the token uses gives way"
    );
    let mut too_few_slots = vec![[0xFF_u8; 4]; 1];
    assert!(
        astro_expert_cache
            .read_route_through_table(
                &runtime,
                0,
                &plan,
                &assignments(&[0, 1]),
                2,
                &filler,
                &mut too_few_slots,
            )
            .expect("the lookup runs")
            .err()
            == Some(TableRefusal::TableUnavailable),
        "a slot for every assignment or none"
    );
}

#[test]
#[file_serial]
fn should_fail_and_keep_the_held_experts_when_a_read_fails() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    // Two slots, both taken, no room to grow: expert 1 gives way, and expert
    // 9 lies past the end of a four-expert file, so its read fails.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let mut table_slots = vec![[0xFF_u8; 4]; 2];

    let read_through = astro_expert_cache.read_route_through_table(
        &runtime,
        0,
        &plan,
        &assignments(&[0, 9]),
        2,
        &filler,
        &mut table_slots,
    );

    assert!(read_through.is_err());
    assert_eq!(
        astro_expert_cache
            .routed_expert_coverage(0, &[0, 9])
            .missing_expert_ids,
        vec![9],
        "no token may multiply from a slot the read left unfilled"
    );
    assert_eq!(
        words_held_for(&astro_expert_cache, 0),
        expected_words_for(0)
    );
}

#[test]
#[file_serial]
fn should_grow_a_full_table_between_tokens_so_the_next_lookup_is_served() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    // Two slots taken and four allowed: a lookup would refuse expert 3.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);

    assert!(
        astro_expert_cache
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
            .expect("the table grows")
    );

    let mut table_slots = vec![[0xFF_u8; 4]; 2];
    let read_through = astro_expert_cache
        .read_route_through_table(
            &runtime,
            0,
            &plan,
            &assignments(&[0, 3]),
            4,
            &filler,
            &mut table_slots,
        )
        .expect("the lookup runs")
        .expect("the grown table serves the route");
    assert_eq!(read_through.missing_expert_count, 1);
    for expert_id in [0, 1, 3] {
        assert_eq!(
            words_held_for(&astro_expert_cache, expert_id),
            expected_words_for(expert_id),
            "expert {expert_id} is in the grown table with its own bytes"
        );
    }
}

#[test]
#[file_serial]
fn should_leave_a_table_alone_that_has_room_or_may_not_grow_or_does_not_exist() {
    let runtime = test_runtime();
    // Two slots taken, and two is all the reply's plan allows.
    let mut full_table_at_its_limit = cache_holding_experts_zero_and_one(&runtime, 2);
    assert!(
        !full_table_at_its_limit
            .grow_table_ahead_of_lookups(&runtime, 0, 2, 2)
            .expect("nothing to build")
    );
    // Four allowed, and already grown to four: two slots are free.
    let mut table_with_room = cache_holding_experts_zero_and_one(&runtime, 4);
    assert!(
        table_with_room
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
            .expect("the table grows")
    );
    assert!(
        !table_with_room
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
            .expect("nothing to build"),
        "a table with a route's worth of free slots stays as it is"
    );
    let mut empty_cache = AstroExpertCache::new(1);
    empty_cache.set_capacity(1024 * 1024 * 1024);
    assert!(
        !empty_cache
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
            .expect("nothing to build")
    );
    // A ceiling the table already fills: growth is refused, as a fetch's is.
    let mut cache_at_its_ceiling = cache_holding_experts_zero_and_one(&runtime, 4);
    cache_at_its_ceiling.capacity_bytes = cache_at_its_ceiling.resident_payload_bytes;
    assert!(
        !cache_at_its_ceiling
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
            .expect("nothing to build")
    );
}

#[test]
#[file_serial]
fn should_report_a_tables_slot_count_and_none_without_a_table() {
    let runtime = test_runtime();
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    assert_eq!(astro_expert_cache.table_slot_count(0), Some(2));
    assert_eq!(
        astro_expert_cache
            .partly_held_table(0, 4)
            .expect("two of four experts are held")
            .slot_count,
        2
    );
    astro_expert_cache
        .grow_table_ahead_of_lookups(&runtime, 0, 4, 2)
        .expect("the table grows");
    assert_eq!(
        astro_expert_cache.table_slot_count(0),
        Some(4),
        "a grown table is another table to a graph built before it grew"
    );
    assert_eq!(astro_expert_cache.table_slot_count(1), None);
    assert!(
        astro_expert_cache.partly_held_table(0, 2).is_none(),
        "a table that holds its whole layer needs no lookup"
    );
}

#[test]
#[file_serial]
fn should_serve_a_block_of_rows_reading_each_missing_expert_once() {
    let ExpertStorage {
        runtime,
        plan,
        filler,
        expert_file: _expert_file,
    } = expert_storage();
    // A drafted block: two rows of two experts each, as one route. Both
    // rows route to expert 3 and each to one more; experts 2 and 3 are
    // missing. Room for two routes is grown before the block, as the model
    // does between rounds.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    assert!(
        astro_expert_cache
            .grow_table_ahead_of_lookups(&runtime, 0, 4, 4)
            .expect("the table grows for two routes")
    );
    let mut table_slots = vec![[0xFF_u8; 4]; 4];

    let read_through = astro_expert_cache
        .read_route_through_table(
            &runtime,
            0,
            &plan,
            &assignments(&[0, 3, 3, 2]),
            4,
            &filler,
            &mut table_slots,
        )
        .expect("the lookup runs")
        .expect("the table serves both rows");

    assert_eq!(read_through.retained_expert_count, 1);
    assert_eq!(
        read_through.missing_expert_count, 2,
        "expert 3 is read once for both rows"
    );
    let slots = slots_written(&table_slots);
    assert_eq!(slots[1], slots[2], "both rows share expert 3's slot");
    for expert_id in [0, 2, 3] {
        assert_eq!(
            words_held_for(&astro_expert_cache, expert_id),
            expected_words_for(expert_id),
            "expert {expert_id} is in the table with its own bytes"
        );
    }
}
