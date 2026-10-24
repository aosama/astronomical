use super::last_prefill_chunk_demand_weight;

#[test]
fn should_scale_last_prefill_chunk_demand_by_earlier_token_density() {
    assert_eq!(last_prefill_chunk_demand_weight(6, 3), 2);
    assert_eq!(last_prefill_chunk_demand_weight(5, 3), 1);
    assert_eq!(last_prefill_chunk_demand_weight(0, 8), 1);
    assert_eq!(last_prefill_chunk_demand_weight(8, 0), 1);
    assert_eq!(last_prefill_chunk_demand_weight(u64::MAX, 1), u64::MAX);
}
