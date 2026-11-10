//! Routed-expert page reads through MLX from the SSD, not the file cache:
//! stock safetensors against the expert-major Atlas layout, at the lowest
//! MLX boundary (load + materialize), with no model math, engine or REST in
//! the way.
//!
//! Every measured page is uncached (checked with `mincore` just before it is
//! read) and is measured under exactly one read shape, because a stock read
//! warms the file cache for every later read of the same bytes. The one
//! deliberate exception is the warm stock shape: the same stock page read
//! again at once, which is what a page the file cache still holds costs.
//!
//! Run it with `cargo measure-cold-expert-page-reads` (see `.cargo/config.toml`).

mod atlas_conversion;
mod cold_page_plan;
mod page_read_shapes;
mod page_timing_report;
mod progress_report;
mod uncached_file_io;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use astronomical_model_serving::{QuantizedExpertLayerPlan, build_quantized_expert_layer_plan};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;
use serial_test::serial;

use super::large_sparse_moe_fixture::{
    StreamingProbeFixture, installed_large_sparse_moe_streaming_fixture, probe_runtime,
};
use atlas_conversion::{AtlasTestFile, write_atlas_test_file};
use cold_page_plan::{
    PlannedPage, ResidencyChecker, cold_stock_expert_ids_by_layer, coldest_layer_indexes,
    plan_cold_pages,
};
use page_read_shapes::{
    load_atlas_page_by_scatter_reads, load_atlas_page_split_on_gpu, load_atlas_records,
    load_stock_page,
};
use page_timing_report::{ShapeTimings, report_shape_timings};

const ATLAS_LAYER_COUNT: usize = 16;
const PROBE_WALL_CLOCK_CEILING: Duration = Duration::from_secs(120);
const MINIMUM_COLD_PAGES_PER_SHAPE: usize = 10;

/// The read shapes that need a cold page each, rotated across pages.
#[derive(Clone, Copy, Debug)]
enum ColdReadShape {
    StockProductionLoad,
    AtlasRecordsOnly,
    AtlasRecordsSplitOnGpu,
    AtlasScatterReads,
}

const COLD_READ_SHAPES: [ColdReadShape; 4] = [
    ColdReadShape::StockProductionLoad,
    ColdReadShape::AtlasRecordsOnly,
    ColdReadShape::AtlasRecordsSplitOnGpu,
    ColdReadShape::AtlasScatterReads,
];

const REPORTED_SHAPE_NAMES: [&str; 5] = [
    "StockProductionLoad",
    "StockProductionLoadWarm",
    "AtlasRecordsOnly",
    "AtlasRecordsSplitOnGpu",
    "AtlasScatterReads",
];

#[test]
#[ignore = "converts installed expert layers to an atlas and times routed pages read from the SSD through MLX"]
#[serial]
fn should_time_cold_routed_pages_through_mlx_for_stock_and_atlas_layouts() {
    assert!(
        !cfg!(debug_assertions),
        "page-read timings need an optimized build (--release)"
    );
    let probe_started_at = Instant::now();
    let fixture = installed_large_sparse_moe_streaming_fixture();
    let runtime = probe_runtime();
    let experts_per_page = fixture.config.experts_per_token() as usize;
    let layer_plans = all_layer_plans(&fixture);
    let mut residency_checker = ResidencyChecker::default();
    let cold_expert_ids_by_layer = cold_stock_expert_ids_by_layer(
        &layer_plans,
        fixture.config.expert_count() as usize,
        &mut residency_checker,
    );
    let atlas_layer_indexes = coldest_layer_indexes(&cold_expert_ids_by_layer, ATLAS_LAYER_COUNT);
    let atlas_directory = tempfile::tempdir().expect("a scratch directory for the atlas");
    let atlas_path = atlas_directory.path().join("experts.atlas");
    let atlas = write_atlas_test_file(&fixture, &atlas_layer_indexes, &atlas_path);
    eprintln!(
        "[atlas-page-reads] atlas layers {atlas_layer_indexes:?}, record {} bytes",
        atlas.record_byte_count
    );

    let planned_pages = plan_cold_pages(
        &layer_plans,
        &cold_expert_ids_by_layer,
        &atlas,
        experts_per_page,
    );
    assert!(
        planned_pages.len() > COLD_READ_SHAPES.len() * MINIMUM_COLD_PAGES_PER_SHAPE,
        "only {} uncached pages remain; this probe needs expert bytes the file cache does not \
         hold (they return once macOS reclaims the cache)",
        planned_pages.len()
    );
    // The first planned page stays unread for the byte-equality check.
    let shape_timings = measure_pages(
        &runtime,
        &atlas,
        &planned_pages[1..],
        &mut residency_checker,
    );
    let page_byte_count = atlas.record_byte_count * experts_per_page as u64;
    report_shape_timings(
        &shape_timings,
        page_byte_count,
        fixture.config.layer_count() as usize,
    );
    assert_atlas_tensors_match_stock(&runtime, &atlas, &planned_pages[0]);

    for shape_timing in &shape_timings {
        assert!(
            shape_timing.page_milliseconds.len() >= MINIMUM_COLD_PAGES_PER_SHAPE,
            "{} measured only {} pages",
            shape_timing.shape_name,
            shape_timing.page_milliseconds.len()
        );
    }
    assert!(
        probe_started_at.elapsed() <= PROBE_WALL_CLOCK_CEILING,
        "the probe exceeded its 120-second ceiling"
    );
}

fn all_layer_plans(fixture: &StreamingProbeFixture) -> Vec<QuantizedExpertLayerPlan> {
    let mut layer_plans = Vec::with_capacity(fixture.config.layer_count() as usize);
    for layer_index in 0..fixture.config.layer_count() as usize {
        let layer_prefix = format!("language_model.model.layers.{layer_index}.mlp");
        layer_plans.push(
            build_quantized_expert_layer_plan(
                &fixture.model_directory,
                &fixture.weight_map,
                &layer_prefix,
                &fixture.config,
            )
            .expect("every sparse layer plan should build"),
        );
    }
    layer_plans
}

/// Shapes rotate per round of layers, so every shape samples every atlas
/// layer rather than always the same few.
fn cold_read_shape_for(planned_page_index: usize, atlas_layer_count: usize) -> ColdReadShape {
    let layer_round = planned_page_index / atlas_layer_count;
    COLD_READ_SHAPES[(planned_page_index + layer_round) % COLD_READ_SHAPES.len()]
}

fn measure_pages(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_pages: &[PlannedPage],
    residency_checker: &mut ResidencyChecker,
) -> Vec<ShapeTimings> {
    let mut page_milliseconds_by_shape: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let mut progress = progress_report::ProgressReporter::new("page-reads", planned_pages.len());
    let measuring_started_at = Instant::now();
    for (planned_page_index, planned_page) in planned_pages.iter().enumerate() {
        progress.record(planned_page_index + 1, measuring_started_at);
        let page_is_cached = residency_checker.stock_page_is_cached(&planned_page.stock_manifest)
            || residency_checker.atlas_page_is_cached(
                atlas,
                planned_page.atlas_layer_position,
                &planned_page.expert_ids,
            );
        if page_is_cached {
            continue;
        }
        let cold_read_shape = cold_read_shape_for(planned_page_index, atlas.layer_indexes.len());
        for (shape_name, page_load_elapsed) in
            measure_one_page(runtime, atlas, planned_page, cold_read_shape)
        {
            page_milliseconds_by_shape
                .entry(shape_name)
                .or_default()
                .push(page_load_elapsed.as_secs_f64() * 1_000.0);
        }
    }
    REPORTED_SHAPE_NAMES
        .into_iter()
        .map(|shape_name| ShapeTimings {
            shape_name,
            page_milliseconds: page_milliseconds_by_shape
                .remove(shape_name)
                .unwrap_or_default(),
        })
        .collect()
}

/// The page under its cold shape; a stock page is then read once more while
/// the file cache holds it, which is the warm stock shape.
fn measure_one_page(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_page: &PlannedPage,
    cold_read_shape: ColdReadShape,
) -> Vec<(&'static str, Duration)> {
    let atlas_layer_position = planned_page.atlas_layer_position;
    let expert_ids = &planned_page.expert_ids;
    match cold_read_shape {
        ColdReadShape::StockProductionLoad => vec![
            (
                "StockProductionLoad",
                load_stock_page(runtime, &planned_page.stock_manifest).page_load_elapsed,
            ),
            (
                "StockProductionLoadWarm",
                load_stock_page(runtime, &planned_page.stock_manifest).page_load_elapsed,
            ),
        ],
        ColdReadShape::AtlasRecordsOnly => vec![(
            "AtlasRecordsOnly",
            load_atlas_records(runtime, atlas, atlas_layer_position, expert_ids).page_load_elapsed,
        )],
        ColdReadShape::AtlasRecordsSplitOnGpu => vec![(
            "AtlasRecordsSplitOnGpu",
            load_atlas_page_split_on_gpu(runtime, atlas, atlas_layer_position, expert_ids)
                .page_load_elapsed,
        )],
        ColdReadShape::AtlasScatterReads => vec![(
            "AtlasScatterReads",
            load_atlas_page_by_scatter_reads(atlas, atlas_layer_position, expert_ids)
                .page_load_elapsed,
        )],
    }
}

/// The atlas must hand the expert math exactly the bytes the stock path does.
fn assert_atlas_tensors_match_stock(
    runtime: &MlxRuntime,
    atlas: &AtlasTestFile,
    planned_page: &PlannedPage,
) {
    let atlas_layer_position = planned_page.atlas_layer_position;
    let stock_tensors = load_stock_page(runtime, &planned_page.stock_manifest).tensors_by_suffix;
    let split_tensors = load_atlas_page_split_on_gpu(
        runtime,
        atlas,
        atlas_layer_position,
        &planned_page.expert_ids,
    )
    .tensors_by_suffix;
    let scattered_tensors =
        load_atlas_page_by_scatter_reads(atlas, atlas_layer_position, &planned_page.expert_ids)
            .tensors_by_suffix;
    for atlas_tensors in [&split_tensors, &scattered_tensors] {
        assert_same_tensors(runtime, &stock_tensors, atlas_tensors);
    }
}

fn assert_same_tensors(
    runtime: &MlxRuntime,
    stock_tensors: &HashMap<String, MlxArray>,
    atlas_tensors: &HashMap<String, MlxArray>,
) {
    assert_eq!(
        stock_tensors.len(),
        atlas_tensors.len(),
        "both layouts present the same tensors"
    );
    for (tensor_suffix, stock_tensor) in stock_tensors {
        let atlas_tensor = &atlas_tensors[tensor_suffix];
        assert_eq!(
            stock_tensor.shape(),
            atlas_tensor.shape(),
            "{tensor_suffix} shape"
        );
        let tensors_are_equal = runtime
            .array_equal(stock_tensor, atlas_tensor, false)
            .expect("array_equal builds");
        assert_eq!(
            tensors_are_equal
                .to_vec_bool()
                .expect("the comparison evaluates"),
            vec![true],
            "{tensor_suffix} bytes differ between layouts"
        );
    }
}
