//! Expert-route predictor contracts: gradient correctness against a
//! brute-force numerical reference, convergence on a learnable pattern,
//! bounded training slices, and accuracy measurement.

use std::time::{Duration, Instant};

use astronomical_model_serving::{
    ExpertRoutePredictor, ExpertRoutePredictorConfig, RouteObservationRecord, RouteObservationRing,
    evaluate_predictor_accuracy, train_predictor_slice,
};

fn test_config() -> ExpertRoutePredictorConfig {
    ExpertRoutePredictorConfig {
        layer_count: 2,
        expert_count: 6,
        vocabulary_size: 32,
        embedding_dim: 4,
        hidden_dim: 5,
        learning_rate: 0.5,
        seed: 7,
    }
}

fn route(expert_ids: &[u16]) -> Option<Vec<u16>> {
    let mut sorted = expert_ids.to_vec();
    sorted.sort_unstable();
    Some(sorted)
}

fn record(
    token_id: u32,
    previous: Option<Vec<Option<Vec<u16>>>>,
    current: Vec<Option<Vec<u16>>>,
) -> RouteObservationRecord {
    RouteObservationRecord {
        input_token_id: token_id,
        previous_token_route: previous,
        token_route: current,
    }
}

/// Central-difference reference for one weight's gradient: recompute the loss
/// with the weight nudged plus and minus epsilon and average the slope.
fn numerical_embedding_gradient(
    predictor: &ExpertRoutePredictor,
    observation: &RouteObservationRecord,
    token_id: u32,
    element_index: usize,
    epsilon: f32,
) -> f32 {
    let mut nudged_plus = predictor.clone();
    let mut nudged_minus = predictor.clone();
    nudged_plus.perturb_embedding_for_tests(token_id, element_index, epsilon);
    nudged_minus.perturb_embedding_for_tests(token_id, element_index, -epsilon);
    (summed_loss(&nudged_plus, observation) - summed_loss(&nudged_minus, observation))
        / (2.0 * epsilon)
}

fn numerical_layer_input_weight_gradient(
    predictor: &ExpertRoutePredictor,
    observation: &RouteObservationRecord,
    layer_index: usize,
    element_index: usize,
    epsilon: f32,
) -> f32 {
    let mut nudged_plus = predictor.clone();
    let mut nudged_minus = predictor.clone();
    nudged_plus.perturb_layer_input_weight_for_tests(layer_index, element_index, epsilon);
    nudged_minus.perturb_layer_input_weight_for_tests(layer_index, element_index, -epsilon);
    (summed_loss(&nudged_plus, observation) - summed_loss(&nudged_minus, observation))
        / (2.0 * epsilon)
}

fn numerical_layer_output_weight_gradient(
    predictor: &ExpertRoutePredictor,
    observation: &RouteObservationRecord,
    layer_index: usize,
    element_index: usize,
    epsilon: f32,
) -> f32 {
    let mut nudged_plus = predictor.clone();
    let mut nudged_minus = predictor.clone();
    nudged_plus.perturb_layer_output_weight_for_tests(layer_index, element_index, epsilon);
    nudged_minus.perturb_layer_output_weight_for_tests(layer_index, element_index, -epsilon);
    (summed_loss(&nudged_plus, observation) - summed_loss(&nudged_minus, observation))
        / (2.0 * epsilon)
}

fn summed_loss(predictor: &ExpertRoutePredictor, observation: &RouteObservationRecord) -> f32 {
    let logits_per_layer = predictor.forward_logits(
        observation.input_token_id,
        observation.previous_token_route.as_deref(),
    );
    let mut total_loss = 0.0_f32;
    for (layer_index, layer_label) in observation.token_route.iter().enumerate() {
        let Some(routed_expert_ids) = layer_label else {
            continue;
        };
        let Some(layer_logits) = logits_per_layer.get(layer_index) else {
            continue;
        };
        let mut routed_expert_id_iter = routed_expert_ids.iter().copied().peekable();
        for (logit_slot, logit) in layer_logits.iter().enumerate() {
            let is_routed = routed_expert_id_iter
                .peek()
                .is_some_and(|next| usize::from(*next) == logit_slot);
            if is_routed {
                routed_expert_id_iter.next();
            }
            let probability = 1.0 / (1.0 + (-logit).exp());
            total_loss -= if is_routed {
                probability.clamp(1.0e-7, 1.0).ln()
            } else {
                (1.0 - probability).clamp(1.0e-7, 1.0).ln()
            };
        }
    }
    total_loss
}

#[test]
fn backward_matches_the_brute_force_numerical_reference() {
    let config = test_config();
    let predictor = ExpertRoutePredictor::new(config);
    let observation = record(
        11,
        Some(vec![route(&[1, 3]), route(&[2])]),
        vec![route(&[0, 4]), route(&[5])],
    );
    let epsilon = 1.0e-3;
    for element_index in [0_usize, 7, 19] {
        let analytic = predictor.input_weight_gradient_for_tests(&observation, 0, element_index);
        let numerical = numerical_layer_input_weight_gradient(
            &predictor,
            &observation,
            0,
            element_index,
            epsilon,
        );
        assert!(
            (analytic - numerical).abs() < 5.0e-3,
            "input-weight gradient element {element_index}: analytic={analytic} numerical={numerical}"
        );
    }
    for element_index in [0_usize, 5, 11] {
        let analytic = predictor.output_weight_gradient_for_tests(&observation, 1, element_index);
        let numerical = numerical_layer_output_weight_gradient(
            &predictor,
            &observation,
            1,
            element_index,
            epsilon,
        );
        assert!(
            (analytic - numerical).abs() < 5.0e-3,
            "output-weight gradient element {element_index}: analytic={analytic} numerical={numerical}"
        );
    }
    let embedding_analytic = predictor.embedding_gradient_for_tests(&observation, 3);
    let embedding_numerical =
        numerical_embedding_gradient(&predictor, &observation, 11, 3, epsilon);
    assert!(
        (embedding_analytic - embedding_numerical).abs() < 5.0e-3,
        "embedding gradient: analytic={embedding_analytic} numerical={embedding_numerical}"
    );
}

#[test]
fn converges_to_near_perfect_top_k_on_a_learnable_pattern() {
    let config = ExpertRoutePredictorConfig {
        layer_count: 1,
        expert_count: 8,
        vocabulary_size: 16,
        embedding_dim: 8,
        hidden_dim: 12,
        learning_rate: 0.05,
        seed: 3,
    };
    let mut predictor = ExpertRoutePredictor::new(config);
    // Token k always routes expert k: a pattern the embedding alone can learn.
    let observations: Vec<RouteObservationRecord> = (0..8_u32)
        .map(|token_id| record(token_id, None, vec![route(&[token_id as u16])]))
        .collect();
    for _step in 0..200 {
        for observation in &observations {
            predictor.train_on_record(observation);
        }
    }
    let accuracy = evaluate_predictor_accuracy(&predictor, &observations, 1);
    let overall_hit_rate = accuracy
        .overall_hit_rate()
        .expect("the pattern records measured at least one expert");
    assert!(
        overall_hit_rate > 0.9,
        "a learnable one-expert-per-token pattern must reach near-perfect top-1; hit_rate={overall_hit_rate}"
    );
}

#[test]
fn unlabeled_layers_receive_no_gradient() {
    let config = test_config();
    let predictor = ExpertRoutePredictor::new(config);
    let observation = record(5, None, vec![None, route(&[2])]);
    for element_index in 0..config.hidden_dim * config.head_input_dim() {
        let gradient = predictor.input_weight_gradient_for_tests(&observation, 0, element_index);
        assert_eq!(
            gradient, 0.0,
            "an unlabeled layer's head must receive no gradient; element {element_index}={gradient}"
        );
    }
    let labeled_layer_has_gradient =
        (0..config.expert_count * config.hidden_dim).any(|element_index| {
            predictor.output_weight_gradient_for_tests(&observation, 1, element_index) != 0.0
        });
    assert!(
        labeled_layer_has_gradient,
        "a labeled layer's head must receive a gradient"
    );
}

#[test]
fn zero_budget_consumes_nothing_and_drains_no_ring() {
    let config = test_config();
    let mut predictor = ExpertRoutePredictor::new(config);
    let mut ring = RouteObservationRing::new(8);
    ring.record_observation(record(1, None, vec![route(&[1]), None]));
    let outcome = train_predictor_slice(&mut predictor, &mut ring, Duration::ZERO, Instant::now());
    assert_eq!(outcome.consumed_record_count, 0);
    assert!(!outcome.stopped_for_budget);
    assert_eq!(ring.observation_count(), 1);
}

#[test]
fn exhausted_budget_stops_between_records_and_keeps_the_rest() {
    let config = test_config();
    let mut predictor = ExpertRoutePredictor::new(config);
    let mut ring = RouteObservationRing::new(8);
    for token_id in 0..4_u32 {
        ring.record_observation(record(token_id, None, vec![route(&[1]), None]));
    }
    // A budget already elapsed before the first record stops immediately.
    let elapsed_budget = Duration::from_secs(0);
    let started_at = Instant::now() - Duration::from_secs(1);
    let outcome = train_predictor_slice(
        &mut predictor,
        &mut ring,
        elapsed_budget.max(Duration::from_nanos(1)),
        started_at,
    );
    assert!(outcome.stopped_for_budget);
    assert_eq!(outcome.consumed_record_count, 0);
    assert_eq!(ring.observation_count(), 4);
}

#[test]
fn a_slice_drains_the_ring_oldest_first_when_time_remains() {
    let config = test_config();
    let mut predictor = ExpertRoutePredictor::new(config);
    let mut ring = RouteObservationRing::new(8);
    for token_id in 0..4_u32 {
        ring.record_observation(record(
            token_id,
            None,
            vec![route(&[token_id as u16]), None],
        ));
    }
    let outcome = train_predictor_slice(
        &mut predictor,
        &mut ring,
        Duration::from_secs(60),
        Instant::now(),
    );
    assert_eq!(outcome.consumed_record_count, 4);
    assert!(!outcome.stopped_for_budget);
    assert_eq!(ring.observation_count(), 0);
    assert!(outcome.summed_loss_millis.is_finite() && outcome.summed_loss_millis > 0.0);
}

#[test]
fn accuracy_reports_per_layer_rates_and_handles_unmeasured_layers() {
    let config = test_config();
    let predictor = ExpertRoutePredictor::new(config);
    let records = vec![record(9, None, vec![route(&[0, 1]), None])];
    let accuracy = evaluate_predictor_accuracy(&predictor, &records, 2);
    assert_eq!(accuracy.layer_total_counts[0], 2);
    assert_eq!(accuracy.layer_total_counts[1], 0);
    assert!(accuracy.layer_hit_rate(1).is_none());
    assert!(accuracy.overall_hit_rate().is_some());
}

#[test]
fn evaluate_then_train_scores_before_updating() {
    let config = test_config();
    let mut predictor = ExpertRoutePredictor::new(config);
    let observation = record(3, None, vec![route(&[3]), None]);
    let (hit_count, evaluated_count) =
        astronomical_model_serving::evaluate_then_train(&mut predictor, &observation, 1);
    assert_eq!(evaluated_count, 1);
    assert!(hit_count <= 1);
}

#[test]
fn background_owner_trains_without_blocking_the_caller() {
    let Some(owner) = astronomical_model_serving::ExpertRoutePredictorOwner::try_start(1, 8, 16, 1)
    else {
        panic!("the trainer thread should start for a valid sparse geometry");
    };
    for token_id in 0..8_u32 {
        owner.try_submit(record(token_id, None, vec![route(&[token_id as u16])]));
    }
    let wait_started_at = Instant::now();
    while owner.trained_record_count() == 0 && wait_started_at.elapsed() < Duration::from_secs(2) {
        std::thread::yield_now();
    }
    assert!(
        owner.trained_record_count() > 0,
        "the background trainer must consume submitted observations"
    );
    assert!(owner.evaluated_expert_count() > 0);
    let predictor_program_status = owner.program_status();
    assert_eq!(predictor_program_status.pages_avoided_tenths, 0);
    assert!(predictor_program_status.top_k_accuracy_tenths <= 1_000);
}

#[test]
fn packed_head_inputs_are_layer_major() {
    let predictor = ExpertRoutePredictor::new(test_config());
    let packed_head_inputs = predictor.packed_head_inputs(3, None);
    assert_eq!(
        packed_head_inputs.len(),
        test_config().layer_count * test_config().head_input_dim()
    );
}
