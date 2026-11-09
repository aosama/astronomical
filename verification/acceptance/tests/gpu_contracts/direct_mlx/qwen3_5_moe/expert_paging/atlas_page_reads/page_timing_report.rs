//! Summarizes each read shape's page timings: percentiles, effective rate,
//! and what one fully cold token would cost across every sparse layer.

pub(super) struct ShapeTimings {
    pub(super) shape_name: &'static str,
    pub(super) page_milliseconds: Vec<f64>,
}

pub(super) fn report_shape_timings(
    shape_timings: &[ShapeTimings],
    page_byte_count: u64,
    layer_count: usize,
) {
    for shape_timing in shape_timings {
        let mut sorted_milliseconds = shape_timing.page_milliseconds.clone();
        sorted_milliseconds.sort_by(f64::total_cmp);
        if sorted_milliseconds.is_empty() {
            eprintln!(
                "[atlas-page-reads] {}: no cold pages",
                shape_timing.shape_name
            );
            continue;
        }
        let median_milliseconds = percentile(&sorted_milliseconds, 0.5);
        eprintln!(
            "[atlas-page-reads] {}: pages={} p10_ms={:.2} median_ms={median_milliseconds:.2} p90_ms={:.2} rate_gb_per_s={:.2} cold_token_all_layers_ms={:.0}",
            shape_timing.shape_name,
            sorted_milliseconds.len(),
            percentile(&sorted_milliseconds, 0.1),
            percentile(&sorted_milliseconds, 0.9),
            page_byte_count as f64 / 1e9 / (median_milliseconds / 1_000.0),
            median_milliseconds * layer_count as f64,
        );
    }
}

/// Nearest-rank percentile of an ascending, non-empty slice.
fn percentile(sorted_milliseconds: &[f64], fraction: f64) -> f64 {
    sorted_milliseconds[((sorted_milliseconds.len() - 1) as f64 * fraction).round() as usize]
}
