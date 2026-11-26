use super::is_identity_slot_order;

#[test]
fn should_accept_a_table_holding_each_expert_in_its_own_slot() {
    assert!(is_identity_slot_order(&[Some(0), Some(1), Some(2)]));
}

#[test]
fn should_refuse_a_table_whose_experts_sit_in_other_slots() {
    assert!(!is_identity_slot_order(&[Some(1), Some(0)]));
}

#[test]
fn should_refuse_a_table_with_an_empty_slot() {
    assert!(!is_identity_slot_order(&[Some(0), None]));
}

#[test]
fn should_accept_an_empty_table_as_ordered_since_the_capacity_check_refuses_it() {
    assert!(is_identity_slot_order(&[]));
}
