//! The requantizer alone: the lowest tier, measured first.

use std::time::{Duration, Instant};

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::RangeSearchPolicy;
use astronomical_quantization::module_inventory::ModuleInventory;
use astronomical_quantization::progress::{ProgressReporter, StageProgress};
use astronomical_quantization::requantize::driver::run_requantize;
use astronomical_quantization::requantize::{
    BitPolicy, DEFAULT_SHARD_BYTES, RequantizeTensorPlan, TensorPlanEntry,
};
use serial_test::serial;

use super::support::{fresh_output_directory, journey_runtime, source_model_directory};

const GIBIBYTE: f64 = 1_073_741_824.0;

/// The repository's batch-job reporting cadence, with slack for the time a
/// single large tensor takes between two progress checks.
const MAXIMUM_PROGRESS_GAP: Duration = Duration::from_secs(10);

#[test]
#[serial]
#[ignore = "needs model_directory in the explicit test-run configuration"]
fn should_requantize_the_installed_source_at_uniform_four_bits_and_report_throughput() {
    let runtime = journey_runtime();
    let source = source_model_directory();
    let output_directory = fresh_output_directory("throughput");
    let plan = uniform_four_bit_plan(&source);
    let warmup_plan = RequantizeTensorPlan {
        entries: vec![plan.entries[0].clone()],
    };
    let mut warmup_reporter = TimedProgressReporter::default();
    run_requantize(
        &runtime,
        &source,
        &output_directory.join("warmup"),
        &warmup_plan,
        RangeSearchPolicy::default_policy(),
        DEFAULT_SHARD_BYTES,
        &mut warmup_reporter,
    )
    .expect("the warmup requantization should succeed");
    let mut timed_reporter = TimedProgressReporter::default();

    let measured_started_at = Instant::now();
    let outcome = run_requantize(
        &runtime,
        &source,
        &output_directory.join("measured"),
        &plan,
        RangeSearchPolicy::default_policy(),
        DEFAULT_SHARD_BYTES,
        &mut timed_reporter,
    )
    .expect("the measured requantization should succeed");
    let elapsed_seconds = measured_started_at.elapsed().as_secs_f64();

    let source_bytes = source_weight_bytes(&source);
    eprintln!(
        "[requantize-throughput] MEASURED tensors={} tensors_per_second={:.1} source_gibibytes_per_second={:.2} output_gibibytes={:.2} peak_mlx_gibibytes={:.2} elapsed_seconds={elapsed_seconds:.2}",
        plan.tensor_count(),
        plan.tensor_count() as f64 / elapsed_seconds,
        source_bytes as f64 / GIBIBYTE / elapsed_seconds,
        outcome.total_output_bytes as f64 / GIBIBYTE,
        outcome.peak_mlx_memory_bytes as f64 / GIBIBYTE
    );
    assert!(timed_reporter.largest_gap() <= MAXIMUM_PROGRESS_GAP);
    let largest_float32_bytes = plan
        .entries
        .iter()
        .fold(0_u64, largest_float32_tensor_bytes);
    assert!(outcome.peak_mlx_memory_bytes <= 8 * largest_float32_bytes);
    let _ = std::fs::remove_dir_all(&output_directory);
}

fn uniform_four_bit_plan(source: &std::path::Path) -> RequantizeTensorPlan {
    let group_size = GroupSize::new(64).expect("valid");
    let inventory =
        ModuleInventory::from_source_directory(source, group_size).expect("the source inventories");
    RequantizeTensorPlan::from_inventory(
        &inventory,
        &BitPolicy::Uniform {
            bit_width: BitWidth::new(4).expect("valid"),
            group_size,
        },
    )
}

fn largest_float32_tensor_bytes(largest_so_far: u64, entry: &TensorPlanEntry) -> u64 {
    largest_so_far.max(entry.shape.iter().product::<usize>() as u64 * 4)
}

/// The bytes of every top-level shard, the data the requantizer reads.
fn source_weight_bytes(source: &std::path::Path) -> u64 {
    let mut source_bytes = 0;
    for directory_entry in std::fs::read_dir(source)
        .expect("the source is listable")
        .flatten()
    {
        let entry_path = directory_entry.path();
        if entry_path.extension().and_then(std::ffi::OsStr::to_str) == Some("safetensors") {
            if let Ok(shard_metadata) = directory_entry.metadata() {
                source_bytes += shard_metadata.len();
            }
        }
    }
    source_bytes
}

/// Records when each progress line arrived, to hold the five-second rule.
#[derive(Default)]
struct TimedProgressReporter {
    report_instants: Vec<Instant>,
}

impl TimedProgressReporter {
    fn largest_gap(&self) -> Duration {
        let mut largest_gap = Duration::ZERO;
        for adjacent_reports in self.report_instants.windows(2) {
            largest_gap = largest_gap.max(adjacent_reports[1] - adjacent_reports[0]);
        }
        largest_gap
    }
}

impl ProgressReporter for TimedProgressReporter {
    fn report(&mut self, progress: &StageProgress) {
        self.report_instants.push(Instant::now());
        eprintln!(
            "[{}] {}/{} ({:.0}%) eta {:?} {}",
            progress.stage_name,
            progress.completed_work_units,
            progress.total_work_units,
            progress.percent_complete(),
            progress.estimate,
            progress.detail
        );
    }
}
