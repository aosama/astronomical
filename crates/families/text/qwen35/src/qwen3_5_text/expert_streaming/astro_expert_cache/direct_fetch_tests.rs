//! Direct-MLX contracts of a fetch straight into a table: the missing
//! experts land in slots planned as an insert plans them, experts the token
//! found are never evicted for them, and a fetch that cannot complete leaves
//! the table without the experts it could not read.

use runtime_integration::MlxRuntime;
use serial_test::file_serial;

use super::*;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_text::expert_streaming::ExpertSlotStorageFiller;
use crate::qwen3_5_text::expert_streaming::expert_slot_storage_fill_tests::{
    EXPERT_COUNT, FileLayout, TENSOR_NAMES, layer_plan, table_words, tagged_word, test_runtime,
    write_expert_file,
};

/// A page already in memory holding `expert_ids`, tagged as the expert file
/// tags them, so a table seeded from it and a slot read from storage can be
/// told apart only by which expert they hold.
fn tagged_page(runtime: &MlxRuntime, expert_ids: &[usize]) -> Qwen3_5PagedExpertWeights {
    let mut table_arrays = Vec::with_capacity(TENSOR_NAMES.len());
    for (tensor_index, (_, _, word_count)) in TENSOR_NAMES.iter().enumerate() {
        let mut words = Vec::with_capacity(expert_ids.len() * word_count);
        for &expert_id in expert_ids {
            words.extend(vec![tagged_word(tensor_index, expert_id); *word_count]);
        }
        table_arrays.push(
            runtime
                .array_from_u32(&words, &[expert_ids.len() as i32, *word_count as i32])
                .expect("the page array builds"),
        );
    }
    let mut table_arrays = table_arrays.into_iter();
    let no_rotation = jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation;
    Qwen3_5PagedExpertWeights {
        gate_projection: next_projection(&mut table_arrays),
        up_projection: next_projection(&mut table_arrays),
        down_projection: next_projection(&mut table_arrays),
        gate_activation_rotation: no_rotation,
        up_activation_rotation: no_rotation,
        down_activation_rotation: no_rotation,
    }
}

/// The next weight, scales and biases of `table_arrays` as one projection.
fn next_projection(
    table_arrays: &mut std::vec::IntoIter<mlx_bridge::MlxArray>,
) -> Qwen3_5AffineWeights {
    Qwen3_5AffineWeights::Quantized {
        packed_weight: table_arrays.next().expect("nine arrays"),
        quantization_scales: table_arrays.next().expect("nine arrays"),
        quantization_biases: table_arrays.next().expect("nine arrays"),
        quantization_bits: 4,
        quantization_group_size: 64,
    }
}

/// An `astro_expert_cache` whose layer 0 table holds experts 0 and 1,
/// inserted the copying way with room to grow to `warm_slot_count`.
pub(super) fn cache_holding_experts_zero_and_one(
    runtime: &MlxRuntime,
    warm_slot_count: usize,
) -> AstroExpertCache {
    let mut astro_expert_cache = AstroExpertCache::new(1);
    astro_expert_cache.set_capacity(1024 * 1024 * 1024);
    astro_expert_cache
        .queue_pending_routed_expert_insert(
            0,
            &[0, 1],
            &tagged_page(runtime, &[0, 1]),
            warm_slot_count,
        )
        .expect("the seed insert queues");
    astro_expert_cache
        .flush_pending_inserts(runtime)
        .expect("the seed insert flushes");
    astro_expert_cache
}

/// The words the table holds for `expert_id`, one entry per tensor, read
/// through the page a token would multiply from.
pub(super) fn words_held_for(astro_expert_cache: &AstroExpertCache, expert_id: usize) -> Vec<u32> {
    let (page, manifest) = astro_expert_cache
        .full_coverage_page(0, &[expert_id], EXPERT_COUNT)
        .expect("the table holds the expert");
    let slot = manifest.page_slot_by_global_expert_id[expert_id] as usize;
    let mut first_word_by_tensor = Vec::with_capacity(TENSOR_NAMES.len());
    for (tensor_index, tensor_words) in table_words(&page).iter().enumerate() {
        first_word_by_tensor.push(tensor_words[slot * TENSOR_NAMES[tensor_index].2]);
    }
    first_word_by_tensor
}

pub(super) fn expected_words_for(expert_id: usize) -> Vec<u32> {
    let mut expected_words = Vec::with_capacity(TENSOR_NAMES.len());
    for tensor_index in 0..TENSOR_NAMES.len() {
        expected_words.push(tagged_word(tensor_index, expert_id));
    }
    expected_words
}

#[test]
#[file_serial]
fn should_read_a_missing_expert_into_the_table_beside_the_ones_it_holds() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler =
        ExpertSlotStorageFiller::open(std::slice::from_ref(&plan)).expect("the filler opens");
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    let fetch_report = astro_expert_cache
        .fetch_missing_experts_into_table(&runtime, 0, &plan, &[3], &[0, 1], 4, &filler)
        .expect("the fetch runs")
        .expect("the table takes the expert");
    assert_eq!(fetch_report.storage_read_count, 1);
    for expert_id in [0, 1, 3] {
        assert_eq!(
            words_held_for(&astro_expert_cache, expert_id),
            expected_words_for(expert_id),
            "expert {expert_id} is in the table with its own bytes"
        );
    }
}

#[test]
#[file_serial]
fn should_evict_an_expert_the_token_does_not_use_when_the_table_is_full() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler =
        ExpertSlotStorageFiller::open(std::slice::from_ref(&plan)).expect("the filler opens");
    // Two slots, both taken, no room to grow.
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let fetch_report = astro_expert_cache
        .fetch_missing_experts_into_table(&runtime, 0, &plan, &[3], &[0], 2, &filler)
        .expect("the fetch runs");
    assert!(fetch_report.is_some(), "expert 1 gives way");
    assert_eq!(
        words_held_for(&astro_expert_cache, 0),
        expected_words_for(0)
    );
    assert_eq!(
        words_held_for(&astro_expert_cache, 3),
        expected_words_for(3)
    );
    let coverage = astro_expert_cache.routed_expert_coverage(0, &[0, 1, 3]);
    assert_eq!(
        coverage.missing_expert_ids,
        vec![1],
        "the unprotected expert was evicted"
    );
}

#[test]
#[file_serial]
fn should_refuse_when_every_slot_holds_an_expert_the_token_uses() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler =
        ExpertSlotStorageFiller::open(std::slice::from_ref(&plan)).expect("the filler opens");
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 2);
    let fetch_report = astro_expert_cache
        .fetch_missing_experts_into_table(&runtime, 0, &plan, &[3], &[0, 1], 2, &filler)
        .expect("the fetch runs");
    assert!(fetch_report.is_none());
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
fn should_refuse_a_layer_with_no_table_and_a_fetch_of_nothing() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler =
        ExpertSlotStorageFiller::open(std::slice::from_ref(&plan)).expect("the filler opens");
    let mut empty_cache = AstroExpertCache::new(1);
    empty_cache.set_capacity(1024 * 1024 * 1024);
    assert!(
        empty_cache
            .fetch_missing_experts_into_table(&runtime, 0, &plan, &[3], &[], 4, &filler)
            .expect("the fetch runs")
            .is_none(),
        "a layer with no table keeps the copying path"
    );
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    assert!(
        astro_expert_cache
            .fetch_missing_experts_into_table(&runtime, 0, &plan, &[], &[0, 1], 4, &filler)
            .expect("the fetch runs")
            .is_none(),
        "nothing missing is nothing to fetch"
    );
}

#[test]
#[file_serial]
fn should_give_up_the_slots_of_experts_it_could_not_read() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler =
        ExpertSlotStorageFiller::open(std::slice::from_ref(&plan)).expect("the filler opens");
    let mut astro_expert_cache = cache_holding_experts_zero_and_one(&runtime, 4);
    // Expert 9 lies past the end of a four-expert file: its read fails.
    let fetch_result = astro_expert_cache.fetch_missing_experts_into_table(
        &runtime,
        0,
        &plan,
        &[9],
        &[0, 1],
        4,
        &filler,
    );
    assert!(fetch_result.is_err());
    let coverage = astro_expert_cache.routed_expert_coverage(0, &[0, 1, 9]);
    assert_eq!(coverage.retained_expert_ids, vec![0, 1]);
    assert_eq!(
        coverage.missing_expert_ids,
        vec![9],
        "no token may multiply from a slot the read left unfilled"
    );
    assert_eq!(
        words_held_for(&astro_expert_cache, 0),
        expected_words_for(0)
    );
}
