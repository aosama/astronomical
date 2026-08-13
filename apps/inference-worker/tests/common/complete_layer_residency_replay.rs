//! Test-only offline replay for complete-layer residency policy research.

use std::cmp::Ordering;

use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompleteLayerEvidence {
    pub(crate) layer_index: usize,
    pub(crate) complete_layer_payload_bytes: u64,
    pub(crate) source_demand_bytes: u64,
    pub(crate) page_readiness_wait_nanoseconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompleteLayerReplayOutcome {
    pub(crate) policy_name: &'static str,
    pub(crate) selected_layer_indices: Vec<usize>,
    pub(crate) retained_payload_bytes: u64,
    pub(crate) avoided_source_demand_bytes: u64,
    pub(crate) avoided_page_readiness_wait_nanoseconds: u64,
}

pub(crate) fn evidence_from_generation_report(
    generation_attribution_report: &Value,
) -> Vec<CompleteLayerEvidence> {
    let phase_reports = generation_attribution_report["expert_source_by_layer"]
        .as_array()
        .expect("generation attribution should report expert source evidence");
    let maximum_layer_index = phase_reports
        .iter()
        .filter_map(|phase_report| phase_report["layer_index"].as_u64())
        .max()
        .expect("expert source evidence should contain a layer index")
        as usize;
    let phase_readiness_rate = |request_phase: &str| {
        let (loaded_bytes, readiness_nanoseconds) = phase_reports
            .iter()
            .filter(|phase_report| phase_report["request_phase"] == request_phase)
            .fold((0_u64, 0_u64), |(bytes, nanoseconds), phase_report| {
                (
                    bytes.saturating_add(
                        phase_report["logical_source_payload_bytes"]
                            .as_u64()
                            .unwrap_or(0),
                    ),
                    nanoseconds.saturating_add(
                        phase_report["page_readiness_wait_nanoseconds"]
                            .as_u64()
                            .unwrap_or(0),
                    ),
                )
            });
        (loaded_bytes, readiness_nanoseconds)
    };
    let prefill_readiness_rate = phase_readiness_rate("prefill");
    let decode_readiness_rate = phase_readiness_rate("decode");
    (0..=maximum_layer_index)
        .map(|layer_index| {
            let mut complete_layer_payload_bytes = 0_u64;
            let mut source_demand_bytes = 0_u64;
            let mut page_readiness_wait_nanoseconds = 0_u64;
            for phase_report in phase_reports.iter().filter(|phase_report| {
                phase_report["layer_index"].as_u64() == Some(layer_index as u64)
            }) {
                let request_phase = phase_report["request_phase"].as_str().unwrap_or("");
                if matches!(request_phase, "prefill" | "retention_transition") {
                    complete_layer_payload_bytes = complete_layer_payload_bytes.max(
                        phase_report["maximum_source_page_payload_bytes"]
                            .as_u64()
                            .unwrap_or(0),
                    );
                }
                if matches!(request_phase, "prefill" | "decode") {
                    let loaded_source_bytes = phase_report["logical_source_payload_bytes"]
                        .as_u64()
                        .unwrap_or(0);
                    let avoided_source_bytes = phase_report["avoided_source_payload_bytes"]
                        .as_u64()
                        .unwrap_or(0);
                    source_demand_bytes = source_demand_bytes
                        .saturating_add(loaded_source_bytes)
                        .saturating_add(avoided_source_bytes);
                    let measured_readiness_nanoseconds =
                        phase_report["page_readiness_wait_nanoseconds"]
                            .as_u64()
                            .unwrap_or(0);
                    let readiness_rate = if request_phase == "prefill" {
                        prefill_readiness_rate
                    } else {
                        decode_readiness_rate
                    };
                    page_readiness_wait_nanoseconds = page_readiness_wait_nanoseconds
                        .saturating_add(measured_readiness_nanoseconds)
                        .saturating_add(estimated_readiness_nanoseconds(
                            avoided_source_bytes,
                            readiness_rate,
                        ));
                }
            }
            assert!(complete_layer_payload_bytes > 0);
            CompleteLayerEvidence {
                layer_index,
                complete_layer_payload_bytes,
                source_demand_bytes,
                page_readiness_wait_nanoseconds,
            }
        })
        .collect()
}

fn estimated_readiness_nanoseconds(
    avoided_source_payload_bytes: u64,
    (measured_source_payload_bytes, measured_readiness_nanoseconds): (u64, u64),
) -> u64 {
    if measured_source_payload_bytes == 0 {
        return 0;
    }
    u64::try_from(
        u128::from(avoided_source_payload_bytes)
            .saturating_mul(u128::from(measured_readiness_nanoseconds))
            / u128::from(measured_source_payload_bytes),
    )
    .unwrap_or(u64::MAX)
}

pub(crate) fn replay_complete_layer_policies(
    layer_evidence: &[CompleteLayerEvidence],
    retained_payload_capacity_bytes: u64,
) -> Vec<CompleteLayerReplayOutcome> {
    vec![
        select_prefix(layer_evidence, retained_payload_capacity_bytes),
        select_by_density(
            "source_bytes_per_retained_byte",
            layer_evidence,
            retained_payload_capacity_bytes,
            |evidence| evidence.source_demand_bytes,
        ),
        select_by_density(
            "readiness_wait_per_retained_byte",
            layer_evidence,
            retained_payload_capacity_bytes,
            |evidence| evidence.page_readiness_wait_nanoseconds,
        ),
        select_readiness_oracle(layer_evidence, retained_payload_capacity_bytes),
    ]
}

fn select_prefix(
    layer_evidence: &[CompleteLayerEvidence],
    retained_payload_capacity_bytes: u64,
) -> CompleteLayerReplayOutcome {
    let mut selected_layers = Vec::new();
    let mut retained_payload_bytes = 0_u64;
    for layer in layer_evidence {
        let next_payload_bytes = retained_payload_bytes
            .checked_add(layer.complete_layer_payload_bytes)
            .unwrap_or(u64::MAX);
        if next_payload_bytes > retained_payload_capacity_bytes {
            break;
        }
        retained_payload_bytes = next_payload_bytes;
        selected_layers.push(*layer);
    }
    outcome("ascending_prefix", &selected_layers)
}

fn select_by_density(
    policy_name: &'static str,
    layer_evidence: &[CompleteLayerEvidence],
    retained_payload_capacity_bytes: u64,
    value: impl Fn(&CompleteLayerEvidence) -> u64,
) -> CompleteLayerReplayOutcome {
    let mut ranked_layers = layer_evidence.to_vec();
    ranked_layers.sort_by(|left, right| {
        compare_density(
            value(right),
            right.complete_layer_payload_bytes,
            value(left),
            left.complete_layer_payload_bytes,
        )
        .then_with(|| left.layer_index.cmp(&right.layer_index))
    });
    let mut selected_layers = Vec::new();
    let mut retained_payload_bytes = 0_u64;
    for layer in ranked_layers {
        if retained_payload_bytes.saturating_add(layer.complete_layer_payload_bytes)
            <= retained_payload_capacity_bytes
        {
            retained_payload_bytes =
                retained_payload_bytes.saturating_add(layer.complete_layer_payload_bytes);
            selected_layers.push(layer);
        }
    }
    selected_layers.sort_by_key(|layer| layer.layer_index);
    outcome(policy_name, &selected_layers)
}

fn compare_density(
    left_value: u64,
    left_payload_bytes: u64,
    right_value: u64,
    right_payload_bytes: u64,
) -> Ordering {
    u128::from(left_value)
        .saturating_mul(u128::from(right_payload_bytes))
        .cmp(&u128::from(right_value).saturating_mul(u128::from(left_payload_bytes)))
}

fn select_readiness_oracle(
    layer_evidence: &[CompleteLayerEvidence],
    retained_payload_capacity_bytes: u64,
) -> CompleteLayerReplayOutcome {
    #[derive(Clone)]
    struct ReplayState {
        retained_payload_bytes: u64,
        avoided_readiness_nanoseconds: u64,
        selected_layers: Vec<CompleteLayerEvidence>,
    }
    let mut frontier = vec![ReplayState {
        retained_payload_bytes: 0,
        avoided_readiness_nanoseconds: 0,
        selected_layers: Vec::new(),
    }];
    for layer in layer_evidence {
        let mut candidates = frontier.clone();
        candidates.extend(frontier.iter().filter_map(|state| {
            let retained_payload_bytes = state
                .retained_payload_bytes
                .checked_add(layer.complete_layer_payload_bytes)?;
            if retained_payload_bytes > retained_payload_capacity_bytes {
                return None;
            }
            let mut selected_layers = state.selected_layers.clone();
            selected_layers.push(*layer);
            Some(ReplayState {
                retained_payload_bytes,
                avoided_readiness_nanoseconds: state
                    .avoided_readiness_nanoseconds
                    .saturating_add(layer.page_readiness_wait_nanoseconds),
                selected_layers,
            })
        }));
        candidates.sort_by_key(|state| state.retained_payload_bytes);
        frontier = Vec::new();
        let mut highest_value_seen = None;
        for candidate in candidates {
            if highest_value_seen
                .is_some_and(|value| value >= candidate.avoided_readiness_nanoseconds)
            {
                continue;
            }
            highest_value_seen = Some(candidate.avoided_readiness_nanoseconds);
            frontier.push(candidate);
        }
    }
    let best_state = frontier
        .into_iter()
        .max_by_key(|state| state.avoided_readiness_nanoseconds)
        .expect("the empty replay state should remain available");
    outcome("readiness_oracle", &best_state.selected_layers)
}

fn outcome(
    policy_name: &'static str,
    selected_layers: &[CompleteLayerEvidence],
) -> CompleteLayerReplayOutcome {
    CompleteLayerReplayOutcome {
        policy_name,
        selected_layer_indices: selected_layers
            .iter()
            .map(|layer| layer.layer_index)
            .collect(),
        retained_payload_bytes: selected_layers.iter().fold(0_u64, |total, layer| {
            total.saturating_add(layer.complete_layer_payload_bytes)
        }),
        avoided_source_demand_bytes: selected_layers.iter().fold(0_u64, |total, layer| {
            total.saturating_add(layer.source_demand_bytes)
        }),
        avoided_page_readiness_wait_nanoseconds: selected_layers
            .iter()
            .fold(0_u64, |total, layer| {
                total.saturating_add(layer.page_readiness_wait_nanoseconds)
            }),
    }
}

#[test]
fn should_find_an_exact_readiness_winner_that_a_greedy_prefix_misses() {
    let layer_evidence = [evidence(0, 6, 6), evidence(1, 5, 6), evidence(2, 5, 6)];
    let outcomes = replay_complete_layer_policies(&layer_evidence, 10);
    assert_eq!(outcomes[0].selected_layer_indices, vec![0]);
    assert_eq!(outcomes[3].selected_layer_indices, vec![1, 2]);
    assert_eq!(outcomes[3].avoided_page_readiness_wait_nanoseconds, 12);
}

#[test]
fn should_break_equal_density_ties_by_layer_index() {
    let layer_evidence = [evidence(0, 5, 10), evidence(1, 5, 10)];
    let outcomes = replay_complete_layer_policies(&layer_evidence, 5);
    assert_eq!(outcomes[1].selected_layer_indices, vec![0]);
    assert_eq!(outcomes[2].selected_layer_indices, vec![0]);
}

fn evidence(
    layer_index: usize,
    complete_layer_payload_bytes: u64,
    page_readiness_wait_nanoseconds: u64,
) -> CompleteLayerEvidence {
    CompleteLayerEvidence {
        layer_index,
        complete_layer_payload_bytes,
        source_demand_bytes: page_readiness_wait_nanoseconds,
        page_readiness_wait_nanoseconds,
    }
}
