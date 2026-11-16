use super::{
    AstroExpertCacheLayerGeometry, AstroExpertCacheTableRules, AstroExpertCacheTables, ExpertKey,
    TableChange,
};

const EXPERT_BYTES: u64 = 10;
const EXPERTS_PER_LAYER: u16 = 8;

fn two_layer_tables(capacity_slot_count: u64, growth_step_slots: usize) -> AstroExpertCacheTables {
    AstroExpertCacheTables::new(
        vec![
            AstroExpertCacheLayerGeometry {
                expert_count: EXPERTS_PER_LAYER,
                expert_payload_bytes: EXPERT_BYTES,
            };
            2
        ],
        capacity_slot_count * EXPERT_BYTES,
        AstroExpertCacheTableRules {
            growth_step_slots,
            idle_window_tokens: 2,
        },
    )
}

fn expert(layer_index: usize, expert_id: u16) -> ExpertKey {
    ExpertKey {
        layer_index,
        expert_id,
    }
}

#[test]
fn should_grow_a_table_by_a_step_and_write_into_it() {
    let mut tables = two_layer_tables(8, 2);

    tables.begin_token();
    let table_changes = tables.serve(0, &[3]);

    assert_eq!(
        table_changes,
        vec![
            TableChange::Grow {
                layer_index: 0,
                slot_capacity: 2,
            },
            TableChange::Write {
                expert: expert(0, 3)
            },
        ]
    );
    assert_eq!(tables.allocated_bytes(), 2 * EXPERT_BYTES);
}

#[test]
fn should_use_a_free_slot_without_reallocating() {
    let mut tables = two_layer_tables(8, 2);
    tables.begin_token();
    tables.serve(0, &[3]);

    tables.begin_token();
    let table_changes = tables.serve(0, &[4]);

    assert_eq!(
        table_changes,
        vec![TableChange::Write {
            expert: expert(0, 4)
        }]
    );
    assert_eq!(tables.reallocation_count(), 1);
}

#[test]
fn should_change_nothing_for_a_route_already_held() {
    let mut tables = two_layer_tables(8, 2);
    tables.begin_token();
    tables.serve(0, &[3]);

    tables.begin_token();
    let table_changes = tables.serve(0, &[3]);

    assert!(table_changes.is_empty());
    assert_eq!(tables.record().read_count(), 1);
}

#[test]
fn should_compact_the_layer_with_idle_experts_to_fund_another_layers_growth() {
    let mut tables = two_layer_tables(8, 2);
    tables.adopt_table(0, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);
    tables.begin_token();
    tables.serve(0, &[0, 1]);

    tables.begin_token();
    let table_changes = tables.serve(1, &[5]);

    assert_eq!(
        table_changes,
        vec![
            TableChange::Compact {
                layer_index: 0,
                slot_capacity: 2,
                dropped_expert_ids: vec![2, 3, 4, 5, 6, 7],
            },
            TableChange::Grow {
                layer_index: 1,
                slot_capacity: 2,
            },
            TableChange::Write {
                expert: expert(1, 5)
            },
        ]
    );
    assert_eq!(tables.held_expert_count(0), 2);
    assert!(tables.is_held(expert(0, 1)));
}

#[test]
fn should_replace_its_own_least_recent_expert_when_nothing_is_idle() {
    let mut tables = two_layer_tables(2, 2);
    tables.begin_token();
    tables.serve(0, &[1]);
    tables.begin_token();
    tables.serve(0, &[2]);

    tables.begin_token();
    let table_changes = tables.serve(0, &[3]);

    assert_eq!(
        table_changes,
        vec![TableChange::Replace {
            expert: expert(0, 3),
            evicted_expert_id: 1,
        }]
    );
    assert_eq!(tables.allocated_bytes(), 2 * EXPERT_BYTES);
}

#[test]
fn should_serve_without_keeping_when_one_token_needs_more_than_fits() {
    let mut tables = two_layer_tables(2, 2);

    tables.begin_token();
    let table_changes = tables.serve(0, &[1, 2, 3]);

    assert_eq!(
        table_changes.last(),
        Some(&TableChange::ServeWithoutKeeping {
            expert: expert(0, 3)
        })
    );
    assert_eq!(tables.held_expert_count(0), 2);
}

#[test]
fn should_hold_nothing_when_capacity_is_zero() {
    let mut tables = two_layer_tables(0, 2);

    tables.begin_token();
    let table_changes = tables.serve(1, &[4]);

    assert_eq!(
        table_changes,
        vec![TableChange::ServeWithoutKeeping {
            expert: expert(1, 4)
        }]
    );
    assert_eq!(tables.allocated_bytes(), 0);
}

#[test]
fn should_not_grow_a_table_past_its_layer() {
    let mut tables = two_layer_tables(32, 16);

    tables.begin_token();
    tables.serve(0, &[0]);

    assert_eq!(tables.slot_capacity(0), usize::from(EXPERTS_PER_LAYER));
}

#[test]
fn should_compact_idle_experts_first_when_capacity_shrinks() {
    let mut tables = two_layer_tables(16, 2);
    tables.adopt_table(0, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);
    tables.adopt_table(1, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);
    tables.begin_token();
    tables.serve(0, &[0, 1, 2, 3, 4, 5, 6, 7]);

    let table_changes = tables.set_capacity(8 * EXPERT_BYTES);

    assert_eq!(
        table_changes,
        vec![TableChange::Compact {
            layer_index: 1,
            slot_capacity: 0,
            dropped_expert_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
        }]
    );
    assert_eq!(tables.allocated_bytes(), 8 * EXPERT_BYTES);
}

#[test]
fn should_give_back_recently_used_experts_when_nothing_is_idle_and_capacity_shrinks() {
    let mut tables = two_layer_tables(4, 2);
    tables.begin_token();
    tables.serve(0, &[1, 2]);
    tables.begin_token();
    tables.serve(1, &[3, 4]);

    tables.set_capacity(2 * EXPERT_BYTES);

    assert_eq!(tables.allocated_bytes(), 2 * EXPERT_BYTES);
    assert_eq!(tables.held_expert_count(0), 0);
    assert_eq!(tables.held_expert_count(1), 2);
}

#[test]
fn should_count_routes_served_from_memory_as_hits_and_keep_them_fresh() {
    let mut tables = two_layer_tables(8, 2);
    tables.adopt_table(0, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);

    tables.begin_token();
    tables.record_served_from_memory(0, &[6, 7]);

    let layer_summary = tables.record().layer_summaries()[0];
    assert_eq!(layer_summary.hit_count, 2);
    assert_eq!(layer_summary.held_never_used_expert_count, 6);
}

#[test]
fn should_not_compact_away_an_expert_a_later_layer_of_the_same_token_uses() {
    let mut tables = two_layer_tables(8, 2);
    tables.adopt_table(1, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);

    tables.begin_token();
    let table_changes = tables.serve_token(&[(0, vec![5]), (1, vec![6])]);

    assert_eq!(
        table_changes,
        vec![
            TableChange::Compact {
                layer_index: 1,
                slot_capacity: 2,
                dropped_expert_ids: vec![0, 1, 2, 3, 4, 5, 7],
            },
            TableChange::Grow {
                layer_index: 0,
                slot_capacity: 2,
            },
            TableChange::Write {
                expert: expert(0, 5)
            },
        ]
    );
    assert!(tables.is_held(expert(1, 6)));
}
