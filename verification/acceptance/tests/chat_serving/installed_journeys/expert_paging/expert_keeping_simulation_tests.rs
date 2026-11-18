use super::{
    KeepingOutcome, ReplyTrace, distinct_expert_count_by_layer, keep_a_fixed_set_of_the_most_used,
    keep_most_recently_used, keep_most_used_so_far, keep_whole_layers_and_recent_tables,
    layers_by_least_repeated_routes, repeated_route_percent_by_layer,
    table_slots_beside_whole_layers,
};

/// Two layers of four experts, 10 bytes each on layer 0 and 20 on layer 1.
/// Layer 0 keeps using experts 0 and 1; layer 1 walks through all four.
fn two_layer_trace() -> ReplyTrace {
    ReplyTrace {
        token_routes: vec![
            vec![vec![0, 1], vec![0, 1]],
            vec![vec![0, 1], vec![2, 3]],
            vec![vec![0, 1], vec![0, 1]],
            vec![vec![0, 1], vec![2, 3]],
        ],
        expert_bytes_by_layer: vec![10, 20],
        experts_per_layer: 4,
    }
}

#[test]
fn should_find_every_assignment_in_memory_when_the_budget_holds_every_expert() {
    let trace = two_layer_trace();
    let everything_bytes = 4 * 10 + 4 * 20;
    for outcome in [
        keep_most_recently_used(&trace, everything_bytes, 2),
        keep_most_used_so_far(&trace, everything_bytes, 2),
        keep_whole_layers_and_recent_tables(&trace, everything_bytes, &[0, 1], 2),
    ] {
        assert_eq!(outcome.scored_token_count, 2);
        assert_eq!(outcome.assignment_count, 8);
        assert_eq!(outcome.missed_assignment_count, 0);
        assert_eq!(outcome.missed_bytes, 0);
        assert_eq!(outcome.found_in_memory_percent(), 100.0);
    }
}

#[test]
fn should_miss_every_assignment_and_count_its_bytes_with_no_budget() {
    let trace = two_layer_trace();
    let outcome = keep_most_recently_used(&trace, 0, 0);
    assert_eq!(outcome.assignment_count, 16);
    assert_eq!(outcome.missed_assignment_count, 16);
    // Eight assignments on each layer: 8 x 10 + 8 x 20.
    assert_eq!(outcome.missed_bytes, 240);
    assert_eq!(outcome.missed_bytes_per_token(), 60.0);
    assert_eq!(outcome.found_in_memory_percent(), 0.0);
}

#[test]
fn should_score_only_the_tokens_from_the_first_scored_one() {
    let trace = two_layer_trace();
    let outcome = keep_most_recently_used(&trace, 0, 3);
    assert_eq!(outcome.scored_token_count, 1);
    assert_eq!(outcome.assignment_count, 4);
}

#[test]
fn should_give_back_the_least_recently_used_expert_first() {
    // Room for layer 0's two experts and two of layer 1's: 2 x 10 + 2 x 20.
    let trace = two_layer_trace();
    let outcome = keep_most_recently_used(&trace, 60, 1);
    // Layer 0 repeats and is always found. Layer 1 alternates between two
    // pairs with room for one pair, so each of its 6 scored assignments misses.
    assert_eq!(outcome.assignment_count, 12);
    assert_eq!(outcome.missed_assignment_count, 6);
    assert_eq!(outcome.missed_bytes, 120);
}

#[test]
fn should_keep_the_most_used_experts_over_the_most_recent_ones() {
    // One layer, room for two experts. Experts 0 and 1 are used every token;
    // 2 and 3 appear once each between them.
    let trace = ReplyTrace {
        token_routes: vec![
            vec![vec![0, 1]],
            vec![vec![0, 1]],
            vec![vec![2, 3]],
            vec![vec![0, 1]],
        ],
        expert_bytes_by_layer: vec![10],
        experts_per_layer: 4,
    };
    let most_used = keep_most_used_so_far(&trace, 20, 3);
    let most_recent = keep_most_recently_used(&trace, 20, 3);
    assert_eq!(
        most_used.missed_assignment_count, 0,
        "0 and 1 were never given back"
    );
    assert_eq!(
        most_recent.missed_assignment_count, 2,
        "2 and 3 pushed them out"
    );
}

#[test]
fn should_serve_a_fixed_set_learned_from_earlier_tokens_without_changing_it() {
    let trace = two_layer_trace();
    // Learned on tokens 0 and 1 with room for layer 0's pair and one layer 1
    // pair: layer 0's experts (used twice each) fill first, then layer 1's
    // lowest-numbered pair among its once-used experts.
    let outcome = keep_a_fixed_set_of_the_most_used(&trace, 60, 0..2, 2);
    assert_eq!(outcome.assignment_count, 8);
    // Token 2 routes layer 1 to 0 and 1 (kept); token 3 to 2 and 3 (not kept).
    assert_eq!(outcome.missed_assignment_count, 2);
    assert_eq!(outcome.missed_bytes, 40);
}

#[test]
fn should_skip_an_expert_too_large_for_what_is_left_and_keep_filling() {
    let trace = two_layer_trace();
    // 30 bytes: layer 0's pair (20) fits, a 20-byte layer 1 expert does not.
    let outcome = keep_a_fixed_set_of_the_most_used(&trace, 30, 0..4, 0);
    assert_eq!(outcome.missed_assignment_count, 8, "all of layer 1 misses");
    assert_eq!(outcome.missed_bytes, 160);
}

#[test]
fn should_split_what_whole_layers_leave_evenly_over_the_other_layers() {
    let trace = two_layer_trace();
    // Layer 0 whole is 40 bytes; 60 more buy three 20-byte slots on layer 1.
    assert_eq!(table_slots_beside_whole_layers(&trace, 100, &[0]), 3);
    // No layer whole: one slot on each layer costs 30 bytes.
    assert_eq!(table_slots_beside_whole_layers(&trace, 100, &[]), 3);
    // More budget than the layers hold: a table never exceeds its layer.
    assert_eq!(table_slots_beside_whole_layers(&trace, 10_000, &[]), 4);
}

#[test]
fn should_give_no_table_slots_when_the_whole_layers_alone_exceed_the_budget() {
    let trace = two_layer_trace();
    assert_eq!(table_slots_beside_whole_layers(&trace, 30, &[0]), 0);
    assert_eq!(table_slots_beside_whole_layers(&trace, 1_000, &[0, 1]), 0);
}

#[test]
fn should_find_whole_layers_always_and_tables_only_for_recent_experts() {
    let trace = two_layer_trace();
    // Layer 0 whole (40) and two slots on layer 1 (40).
    let outcome = keep_whole_layers_and_recent_tables(&trace, 80, &[0], 1);
    assert_eq!(outcome.assignment_count, 12);
    // Layer 1 alternates pairs through a two-slot table: all six miss.
    assert_eq!(outcome.missed_assignment_count, 6);
    assert_eq!(outcome.missed_bytes, 120);
}

#[test]
fn should_measure_how_much_each_layer_repeats_its_recent_routes() {
    let trace = two_layer_trace();
    let repeated_percent = repeated_route_percent_by_layer(&trace, 1);
    // Layer 0: tokens 1 to 3 repeat both experts, token 0 has no past: 6 of 8.
    assert_eq!(repeated_percent[0], 75.0);
    // Layer 1 never repeats the token before it.
    assert_eq!(repeated_percent[1], 0.0);
    // Looking two tokens back, layer 1's tokens 2 and 3 repeat: 4 of 8.
    assert_eq!(repeated_route_percent_by_layer(&trace, 2)[1], 50.0);
    assert_eq!(layers_by_least_repeated_routes(&trace, 1), vec![1, 0]);
}

#[test]
fn should_count_the_different_experts_each_layer_used() {
    assert_eq!(
        distinct_expert_count_by_layer(&two_layer_trace()),
        vec![2, 4]
    );
}

#[test]
fn should_report_zeroes_for_an_empty_trace() {
    let empty_trace = ReplyTrace {
        token_routes: Vec::new(),
        expert_bytes_by_layer: vec![10, 20],
        experts_per_layer: 4,
    };
    let outcome = keep_most_recently_used(&empty_trace, 100, 0);
    assert_eq!(outcome, KeepingOutcome::default());
    assert_eq!(outcome.found_in_memory_percent(), 0.0);
    assert_eq!(outcome.missed_bytes_per_token(), 0.0);
    assert_eq!(
        repeated_route_percent_by_layer(&empty_trace, 4),
        vec![0.0, 0.0]
    );
    assert_eq!(distinct_expert_count_by_layer(&empty_trace), vec![0, 0]);
}
