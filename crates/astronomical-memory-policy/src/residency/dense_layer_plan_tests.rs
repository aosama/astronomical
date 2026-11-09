//! Boundaries of the dense layer residency plan: everything fits, nothing
//! fits, one layer short, exactly at the limit, and below the streaming floor.

use super::dense_layer_plan::{DenseLayerResidencyPlan, DenseLayerResidencyRequirements};

/// Eight layers of 100 bytes, 50 always-resident bytes, 30 working bytes and
/// two slots: 880 bytes fully resident, a 280-byte streaming floor.
const LAYER_BYTES: [u64; 8] = [100; 8];

fn requirements_under(
    active_memory_ceiling_bytes: u64,
) -> DenseLayerResidencyRequirements<'static> {
    DenseLayerResidencyRequirements {
        always_resident_bytes: 50,
        decoder_layer_streamable_bytes: &LAYER_BYTES,
        streaming_slot_count: 2,
        working_memory_bytes: 30,
        active_memory_ceiling_bytes,
    }
}

#[test]
fn should_keep_every_layer_resident_without_slots_when_the_model_fits() {
    let residency_plan = requirements_under(1_000).plan();

    assert_eq!(
        residency_plan,
        DenseLayerResidencyPlan {
            resident_layer_count: 8,
            streaming_slot_bytes: 0,
            streaming_floor_bytes: 280,
            is_below_streaming_floor: false,
        }
    );
}

#[test]
fn should_keep_every_layer_resident_when_the_model_exactly_meets_the_limit() {
    let residency_plan = requirements_under(880).plan();

    assert_eq!(residency_plan.resident_layer_count, 8);
    assert_eq!(residency_plan.streaming_slot_bytes, 0);
}

#[test]
fn should_give_up_three_layers_when_one_byte_short_of_complete_residency() {
    // Streaming starts by paying for two slots, each the size of a layer, so
    // the first streamed plan holds three layers fewer: 879 - 280 = 599
    // bytes keep five of the eight layers.
    let residency_plan = requirements_under(879).plan();

    assert_eq!(residency_plan.resident_layer_count, 5);
    assert_eq!(residency_plan.streaming_slot_bytes, 200);
    assert!(!residency_plan.is_below_streaming_floor);
}

#[test]
fn should_keep_one_layer_resident_when_one_fits_beside_the_slots() {
    let residency_plan = requirements_under(380).plan();

    assert_eq!(residency_plan.resident_layer_count, 1);
}

#[test]
fn should_stream_every_layer_at_exactly_the_streaming_floor() {
    let residency_plan = requirements_under(280).plan();

    assert_eq!(residency_plan.resident_layer_count, 0);
    assert!(!residency_plan.is_below_streaming_floor);
}

#[test]
fn should_stream_every_layer_and_report_a_limit_below_the_streaming_floor() {
    let residency_plan = requirements_under(279).plan();

    assert_eq!(
        residency_plan,
        DenseLayerResidencyPlan {
            resident_layer_count: 0,
            streaming_slot_bytes: 200,
            streaming_floor_bytes: 280,
            is_below_streaming_floor: true,
        }
    );
}

#[test]
fn should_size_the_slots_for_the_largest_layer() {
    let uneven_layer_bytes = [80, 120, 100];
    let residency_plan = DenseLayerResidencyRequirements {
        always_resident_bytes: 0,
        decoder_layer_streamable_bytes: &uneven_layer_bytes,
        streaming_slot_count: 2,
        working_memory_bytes: 0,
        active_memory_ceiling_bytes: 299,
    }
    .plan();

    assert_eq!(residency_plan.streaming_slot_bytes, 240);
    assert_eq!(
        residency_plan.resident_layer_count, 0,
        "59 bytes fit no 80-byte layer"
    );
}

#[test]
fn should_plan_a_model_with_no_decoder_layers_as_fully_resident() {
    let residency_plan = DenseLayerResidencyRequirements {
        always_resident_bytes: 10,
        decoder_layer_streamable_bytes: &[],
        streaming_slot_count: 2,
        working_memory_bytes: 0,
        active_memory_ceiling_bytes: 10,
    }
    .plan();

    assert_eq!(residency_plan.resident_layer_count, 0);
    assert_eq!(residency_plan.streaming_slot_bytes, 0);
}
