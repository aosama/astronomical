use super::decode_seating::{
    DecodeExpertSeating, complete_layers_within_decode_warm_budget,
    seat_and_materialize_complete_layers,
};
use astronomical_memory_policy::{
    phase::MemoryPhase,
    residency::{
        CurrentExpertLayerResidency, ExpertLayerGeometry, RetainedExpertPageClass,
        complete_layer_indexes_required_before_decode, plan_expert_residency,
        publish_request_stable_residency_plan,
    },
};

struct FakeDecodeExpertSeating {
    resident_payload_bytes: Vec<u64>,
    source_read_layer_indexes: Vec<usize>,
    materialized_payload_bytes: u64,
    failing_layer_index: Option<usize>,
    refuses_layer_index: Option<usize>,
    fails_materialization: bool,
}

impl DecodeExpertSeating for FakeDecodeExpertSeating {
    type Error = &'static str;

    fn seat_complete_layer(&mut self, layer_index: usize) -> Result<Option<u64>, Self::Error> {
        if self.failing_layer_index == Some(layer_index) {
            return Err("expert source read failed");
        }
        if self.refuses_layer_index == Some(layer_index) {
            return Ok(None);
        }
        self.source_read_layer_indexes.push(layer_index);
        let missing_payload_bytes = 40 - self.resident_payload_bytes[layer_index];
        self.resident_payload_bytes[layer_index] = 40;
        Ok(Some(missing_payload_bytes))
    }

    fn materialize_complete_layers(&mut self) -> Result<(), Self::Error> {
        if self.fails_materialization {
            return Err("expert materialization failed");
        }
        self.materialized_payload_bytes = self.resident_payload_bytes.iter().sum();
        Ok(())
    }
}

fn partial_prefill_fixture() -> (Vec<ExpertLayerGeometry>, Vec<CurrentExpertLayerResidency>) {
    let geometries = (0..3)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 1,
        })
        .collect();
    let partial_pages = (0..3)
        .map(|layer_index| CurrentExpertLayerResidency {
            layer_index,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 2],
            payload_bytes: 20,
            covered_weighted_demand: 2,
        })
        .collect();
    (geometries, partial_pages)
}

fn fake_seating() -> FakeDecodeExpertSeating {
    FakeDecodeExpertSeating {
        resident_payload_bytes: vec![20; 3],
        source_read_layer_indexes: Vec::new(),
        materialized_payload_bytes: 0,
        failing_layer_index: None,
        refuses_layer_index: None,
        fails_materialization: false,
    }
}

#[test]
fn should_complete_and_materialize_partial_prefill_pages_before_decode() {
    let (geometries, partial_pages) = partial_prefill_fixture();
    let prefill_plan = plan_expert_residency(MemoryPhase::Prefill, 80, &geometries, &partial_pages)
        .expect("prefill must stay below complete fit");
    let (prefill_contract, _) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        None,
        prefill_plan,
        &partial_pages,
        0,
        &geometries,
    );
    let generation_plan = plan_expert_residency(
        MemoryPhase::GenerationPreparation,
        120,
        &geometries,
        &partial_pages,
    )
    .expect("the decode phase fits the complete expert payload");
    let (generation_contract, generation_plan) = publish_request_stable_residency_plan(
        MemoryPhase::GenerationPreparation,
        prefill_contract.as_ref(),
        generation_plan,
        &partial_pages,
        0,
        &geometries,
    );
    assert!(generation_contract.is_none());
    let selected_layer_indexes = complete_layers_within_decode_warm_budget(
        &geometries,
        &partial_pages,
        &complete_layer_indexes_required_before_decode(&generation_plan),
        60,
    );
    assert_eq!(selected_layer_indexes, vec![0, 1, 2]);
    let mut seating = fake_seating();
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &selected_layer_indexes),
        Ok(60)
    );
    assert_eq!(seating.source_read_layer_indexes, vec![0, 1, 2]);
    assert_eq!(seating.materialized_payload_bytes, 120);
    assert_eq!(seating.resident_payload_bytes, vec![40; 3]);
}

#[test]
fn should_charge_only_missing_experts_and_leave_the_unseated_warm_floor() {
    let (geometries, partial_pages) = partial_prefill_fixture();
    assert_eq!(
        complete_layers_within_decode_warm_budget(&geometries, &partial_pages, &[0], 20),
        vec![0],
        "the other layers already retain their warm floor"
    );
    assert!(
        complete_layers_within_decode_warm_budget(&geometries, &partial_pages, &[0], 19).is_empty(),
        "one byte below the incremental completion boundary must refuse seating"
    );
    assert_eq!(
        complete_layers_within_decode_warm_budget(&geometries, &[], &[0, 1, 2], 80),
        vec![0],
        "one complete layer must leave the two-slot warm floor per remaining sparse layer"
    );
    assert!(complete_layers_within_decode_warm_budget(&geometries, &[], &[0], 79).is_empty());
}

#[test]
fn should_not_overcommit_when_the_runtime_grants_less_than_the_plan() {
    let (geometries, partial_pages) = partial_prefill_fixture();
    let selected_layer_indexes =
        complete_layers_within_decode_warm_budget(&geometries, &partial_pages, &[0, 1, 2], 59);
    assert_eq!(selected_layer_indexes, vec![0, 1]);
    let mut seating = fake_seating();
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &selected_layer_indexes),
        Ok(40)
    );
    assert_eq!(seating.materialized_payload_bytes, 100);
    assert_eq!(seating.resident_payload_bytes[2], 20);
}

#[test]
fn should_materialize_successful_seats_even_when_a_later_source_read_fails() {
    let mut seating = fake_seating();
    seating.failing_layer_index = Some(1);
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &[0, 1, 2]),
        Err("expert source read failed")
    );
    assert_eq!(seating.source_read_layer_indexes, vec![0]);
    assert_eq!(seating.materialized_payload_bytes, 80);
}

#[test]
fn should_surface_materialization_failure_and_preserve_a_prior_source_read_failure() {
    let mut seating = fake_seating();
    seating.fails_materialization = true;
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &[0]),
        Err("expert materialization failed")
    );
    seating.failing_layer_index = Some(1);
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &[0, 1]),
        Err("expert source read failed")
    );
}

#[test]
fn should_stop_after_refused_admission_and_materialize_existing_pages_without_new_targets() {
    let mut seating = fake_seating();
    seating.refuses_layer_index = Some(1);
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &[0, 1, 2]),
        Ok(20)
    );
    assert_eq!(seating.source_read_layer_indexes, vec![0]);
    assert_eq!(seating.materialized_payload_bytes, 80);
    seating.materialized_payload_bytes = 0;
    assert_eq!(
        seat_and_materialize_complete_layers(&mut seating, &[]),
        Ok(0)
    );
    assert_eq!(seating.materialized_payload_bytes, 80);
}
