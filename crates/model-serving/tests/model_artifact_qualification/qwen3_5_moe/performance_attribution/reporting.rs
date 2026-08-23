use serde_json::Value;

pub(crate) fn assert_attributed_memory_within_machine_cap(
    phase_name: &str,
    report: &Value,
    machine_derived_metal_cap_bytes: usize,
) {
    let active = report["mlx_active_memory_bytes"]
        .as_u64()
        .expect("successful attribution should report MLX active memory");
    let allocator = report["mlx_allocator_cache_memory_bytes"]
        .as_u64()
        .expect("successful attribution should report MLX allocator-cache memory");
    let observed = active
        .checked_add(allocator)
        .expect("attributed MLX active and allocator-cache memory should not overflow");
    let cap = u64::try_from(machine_derived_metal_cap_bytes)
        .expect("the machine-derived Metal cap should fit in u64");
    assert!(
        observed <= cap,
        "{phase_name} attributed MLX residency {observed} exceeded the machine-derived Metal cap {cap}"
    );
}
pub(crate) fn print_attribution_metadata(phase_name: &str, report: &Value) {
    eprintln!(
        "[performance-attribution] status=report phase={phase_name} prefill_transient_observation_completed={} prefill_observed_transient_high_water_bytes={} resident_model_payload_bytes={} mlx_active_memory_bytes={} mlx_allocator_cache_memory_bytes={} mlx_peak_memory_bytes={}",
        report["prefill_transient_observation_completed"],
        report["prefill_observed_transient_high_water_bytes"],
        report["resident_model_payload_bytes"],
        report["mlx_active_memory_bytes"],
        report["mlx_allocator_cache_memory_bytes"],
        report["mlx_peak_memory_bytes"]
    );
}
pub(crate) fn print_attribution_operation_table(phase_name: &str, report: &Value) {
    let elapsed = report["report_elapsed_nanoseconds"]
        .as_u64()
        .expect("performance-attribution report should include elapsed time");
    let mut rows: Vec<(&str, u64)> = report["operations"]
        .as_array()
        .expect("performance-attribution report should include operation rows")
        .iter()
        .filter_map(|operation| {
            Some((
                operation["operation"].as_str()?,
                operation["total_elapsed_nanoseconds"].as_u64()?,
            ))
        })
        .collect();
    rows.sort_unstable_by(|left, right| right.1.cmp(&left.1));
    for (operation_identifier, total_elapsed_nanoseconds) in rows {
        let percentage = total_elapsed_nanoseconds as f64 / elapsed.max(1) as f64 * 100.0;
        eprintln!(
            "[performance-attribution] status=report phase={phase_name} operation={operation_identifier} total_elapsed_millis={:.3} percent_of_request={percentage:.2}",
            total_elapsed_nanoseconds as f64 / 1_000_000.0
        );
    }
}

pub(crate) fn print_expert_streaming_source_summary_table(phase_name: &str, report: &Value) {
    for source_summary in report["expert_streaming_source_summaries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let Some(streaming_phase) = source_summary["phase"].as_str() else {
            continue;
        };
        let Some(layer_index) = source_summary["layer_index"].as_u64() else {
            continue;
        };
        let source_plan_count = source_summary["source_plan_count"].as_u64().unwrap_or(0);
        let payload_byte_count = source_summary["payload_byte_count"].as_u64().unwrap_or(0);
        eprintln!(
            "[performance-attribution] status=report phase={phase_name} expert_streaming_phase={streaming_phase} layer_index={layer_index} source_plan_count={source_plan_count} source_payload_bytes={payload_byte_count}"
        );
    }
}
