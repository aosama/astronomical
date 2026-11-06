//! What loaded Neural Engine programs really cost in memory, on several real models.
//!
//! The driver wires a program's weights when it runs and releases them 10 to 20
//! seconds after its last use (experiment 2 in `verification/neural-engine`), so a
//! reading taken after a run says little. These journeys sample continuously during
//! prefill and compare the peaks of one loaded model with its programs running and
//! paused, so the difference is the programs alone. Run with no model loaded in either
//! instance.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::NeuralEnginePrefillStatus;
use astronomical_model_serving::Qwen3_5ResidentModel;
use astronomical_neural_engine_research::process_memory::{
    MemorySnapshot, system_reclaimable_bytes,
};
use serial_test::serial;
use tokio::time::timeout;

use super::neural_engine_mlp_prefill::{
    JOURNEY_TIMEOUT, directory_size_gibibytes, load_named_dense_model,
    prefill_only_tokens_per_second, prepare_programs,
};

const JOURNEY_NAME: &str = "neural-engine-program-memory";
const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// Past the driver's 10 to 20 second release, so each arm starts from the same place.
const SETTLE_BEFORE_ARM: Duration = Duration::from_secs(30);
const PASSES_PER_ARM: usize = 2;
/// Long enough to see where the driver's delayed wiring settles.
const PROGRAMS_ALONE_DURATION: Duration = Duration::from_secs(6);
const PROGRAMS_ALONE_REPORT_INTERVAL: Duration = Duration::from_millis(500);
const ROUND_COUNT: usize = 2;
const BYTES_PER_GIBIBYTE: f64 = 1_073_741_824.0;
/// How far what the programs measured at load may sit from what they hold alone. Both
/// read the whole Mac, and they agreed within about 1 percent on every model measured.
const LARGEST_TOLERATED_MEASUREMENT_ERROR: f64 = 0.05;

/// The extremes one arm reached.
#[derive(Clone, Copy, Debug)]
struct MemoryExtremes {
    peak_wired_bytes: u64,
    peak_footprint_bytes: u64,
    lowest_reclaimable_bytes: u64,
}

impl MemoryExtremes {
    fn now() -> Self {
        let snapshot = MemorySnapshot::capture();
        Self {
            peak_wired_bytes: snapshot.system_wired_bytes,
            peak_footprint_bytes: snapshot.process_physical_footprint_bytes,
            lowest_reclaimable_bytes: system_reclaimable_bytes(),
        }
    }

    fn widen_with_current_reading(&mut self) {
        let current = Self::now();
        self.peak_wired_bytes = self.peak_wired_bytes.max(current.peak_wired_bytes);
        self.peak_footprint_bytes = self.peak_footprint_bytes.max(current.peak_footprint_bytes);
        self.lowest_reclaimable_bytes = self
            .lowest_reclaimable_bytes
            .min(current.lowest_reclaimable_bytes);
    }
}

/// Samples on its own thread, so a long forward cannot hide a short peak.
struct MemorySampler {
    should_stop: Arc<AtomicBool>,
    sampling_thread: JoinHandle<MemoryExtremes>,
}

fn sample_until_stopped(should_stop: &AtomicBool, arm_label: &str) -> MemoryExtremes {
    let started_at = Instant::now();
    let mut last_progress_at = started_at;
    let mut extremes = MemoryExtremes::now();
    while !should_stop.load(Ordering::Relaxed) {
        extremes.widen_with_current_reading();
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            last_progress_at = Instant::now();
            eprintln!(
                "[{JOURNEY_NAME}] status=progress arm={arm_label} elapsed_s={:.0} peak_wired_gib={:.2}",
                started_at.elapsed().as_secs_f64(),
                gibibytes(extremes.peak_wired_bytes)
            );
        }
        std::thread::sleep(SAMPLE_INTERVAL);
    }
    extremes
}

impl MemorySampler {
    fn start(arm_label: &'static str) -> Self {
        let should_stop = Arc::new(AtomicBool::new(false));
        let should_stop_for_thread = Arc::clone(&should_stop);
        let sampling_thread =
            std::thread::spawn(move || sample_until_stopped(&should_stop_for_thread, arm_label));
        Self {
            should_stop,
            sampling_thread,
        }
    }

    fn finish(self) -> MemoryExtremes {
        self.should_stop.store(true, Ordering::Relaxed);
        self.sampling_thread.join().expect("the sampler finishes")
    }
}

/// Sits out the driver's release, saying so at the cadence of any other long job.
fn settle_reporting_progress(arm_label: &str) {
    let settling_started_at = Instant::now();
    while settling_started_at.elapsed() < SETTLE_BEFORE_ARM {
        eprintln!(
            "[{JOURNEY_NAME}] status=progress arm={arm_label} phase=settling remaining_s={:.0}",
            SETTLE_BEFORE_ARM
                .saturating_sub(settling_started_at.elapsed())
                .as_secs_f64()
        );
        std::thread::sleep(PROGRESS_INTERVAL);
    }
}

fn gibibytes(byte_count: u64) -> f64 {
    byte_count as f64 / BYTES_PER_GIBIBYTE
}

/// A settled start, then prefill passes with the sampler running.
fn measure_arm(
    model: &Qwen3_5ResidentModel,
    prompt_token_ids: &[u32],
    programs_run: bool,
) -> (MemoryExtremes, MemoryExtremes, f64) {
    let arm_label = if programs_run { "both" } else { "gpu" };
    settle_reporting_progress(arm_label);
    model.set_neural_engine_mlp_prefill_paused(!programs_run);
    model.set_neural_engine_output_gate_paused(!programs_run);
    let settled = MemoryExtremes::now();
    let sampler = MemorySampler::start(arm_label);
    let mut tokens_per_second = 0.0;
    for _ in 0..PASSES_PER_ARM {
        tokens_per_second = prefill_only_tokens_per_second(model, prompt_token_ids);
    }
    (settled, sampler.finish(), tokens_per_second)
}

fn report_arm(
    arm_label: &str,
    settled: MemoryExtremes,
    peaks: MemoryExtremes,
    tokens_per_second: f64,
) {
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED arm={arm_label} tps={tokens_per_second:.0} \
         wired_settled_gib={:.2} wired_peak_gib={:.2} footprint_settled_gib={:.2} footprint_peak_gib={:.2} \
         reclaimable_settled_gib={:.2} reclaimable_lowest_gib={:.2}",
        gibibytes(settled.peak_wired_bytes),
        gibibytes(peaks.peak_wired_bytes),
        gibibytes(settled.peak_footprint_bytes),
        gibibytes(peaks.peak_footprint_bytes),
        gibibytes(settled.lowest_reclaimable_bytes),
        gibibytes(peaks.lowest_reclaimable_bytes),
    );
}

/// What the programs hold alone: settled, then cycled with the GPU idle and nothing of
/// MLX's working set in the reading. The driver wires with a delay, so this reports
/// over time until well after the reading has settled.
fn measure_programs_alone(model: &Qwen3_5ResidentModel) -> u64 {
    settle_reporting_progress("programs-alone");
    let settled_wired_bytes = MemorySnapshot::capture().system_wired_bytes;
    let sampler = MemorySampler::start("programs-alone");
    let cycling_started_at = Instant::now();
    let mut last_report_at = cycling_started_at;
    let mut pass_count = 0_u32;
    while cycling_started_at.elapsed() < PROGRAMS_ALONE_DURATION {
        model
            .run_every_neural_engine_program_once()
            .expect("the programs run");
        pass_count += 1;
        if last_report_at.elapsed() >= PROGRAMS_ALONE_REPORT_INTERVAL {
            last_report_at = Instant::now();
            eprintln!(
                "[{JOURNEY_NAME}] MEASURED programs_alone_s={:.1} passes={pass_count} wired_above_settled_gib={:.3}",
                cycling_started_at.elapsed().as_secs_f64(),
                gibibytes(
                    MemorySnapshot::capture()
                        .system_wired_bytes
                        .saturating_sub(settled_wired_bytes)
                ),
            );
        }
    }
    sampler
        .finish()
        .peak_wired_bytes
        .saturating_sub(settled_wired_bytes)
}

/// macOS's compiled copies for this test binary, which nothing else ever removes and
/// which cost about as much disk as the programs on every rebuild of a test.
fn compiled_copies_of_this_test_binary() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let executable_name = executable.file_name()?;
    let home_directory = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home_directory)
            .join("Library/Caches")
            .join(executable_name),
    )
}

async fn measure_program_memory(dense_model_id: &str) {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let (mut model, prompt_token_ids) = load_named_dense_model(&development_config, dense_model_id);
    let program_directory = tempfile::tempdir().expect("a scratch directory for the programs");
    // The sampler's ticks are the only output during this wait, which includes the
    // build and the idle wait before measuring.
    let preparation_ticker = MemorySampler::start("preparing");
    let status = prepare_programs(&mut model, Some(program_directory.path()));
    preparation_ticker.finish();
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED model={dense_model_id} programs_on_disk_gib={:.2} status={status:?}",
        directory_size_gibibytes(program_directory.path()),
    );
    let NeuralEnginePrefillStatus::Active {
        measured_memory_bytes,
        measured_speed_percent: None,
        ..
    } = status
    else {
        panic!("the programs must load for this measurement: {status:?}");
    };
    let programs_alone_bytes = measure_programs_alone(&model);
    let mut peak_wired_with_programs = 0;
    let mut peak_wired_without_programs = 0;
    for round_index in 0..ROUND_COUNT {
        let both_goes_first = round_index % 2 == 0;
        for programs_run in [both_goes_first, !both_goes_first] {
            let (settled, peaks, tokens_per_second) =
                measure_arm(&model, &prompt_token_ids, programs_run);
            report_arm(
                if programs_run { "both" } else { "gpu" },
                settled,
                peaks,
                tokens_per_second,
            );
            let arm_peak = if programs_run {
                &mut peak_wired_with_programs
            } else {
                &mut peak_wired_without_programs
            };
            *arm_peak = peaks.peak_wired_bytes.max(*arm_peak);
        }
    }
    drop(program_directory);
    if let Some(compiled_copies) = compiled_copies_of_this_test_binary() {
        let _ = std::fs::remove_dir_all(compiled_copies);
    }
    let both_minus_gpu_bytes = peak_wired_with_programs.saturating_sub(peak_wired_without_programs);
    // The surfaces are wired when programs load, before the settled reading started, so
    // only the rest of the cost can be compared with it.
    let surface_bytes = model.neural_engine_program_surface_bytes();
    let measured_wired_bytes = measured_memory_bytes.saturating_sub(surface_bytes);
    let measured_over_alone = measured_wired_bytes as f64 / programs_alone_bytes as f64;
    eprintln!(
        "[{JOURNEY_NAME}] MEASURED model={dense_model_id} measured_at_load_gib={:.2} surfaces_gib={:.2} programs_alone_gib={:.2} both_minus_gpu_gib={:.2} measured_over_alone={measured_over_alone:.3}",
        gibibytes(measured_memory_bytes),
        gibibytes(surface_bytes),
        gibibytes(programs_alone_bytes),
        gibibytes(both_minus_gpu_bytes),
    );
    assert!(
        (measured_over_alone - 1.0).abs() <= LARGEST_TOLERATED_MEASUREMENT_ERROR,
        "the load-time measurement must track what the programs hold: {measured_over_alone:.3}"
    );
}

#[tokio::test]
#[ignore = "loads a real 9B model and its Neural Engine programs"]
#[serial]
async fn should_measure_peak_program_memory_of_the_nine_billion_parameter_model() {
    timeout(
        JOURNEY_TIMEOUT,
        measure_program_memory("Ornith-1.5-9B-MLX-4bit-vision"),
    )
    .await
    .expect("the journey must finish inside its bound");
}

#[tokio::test]
#[ignore = "loads a real 0.8B model and its Neural Engine programs"]
#[serial]
async fn should_measure_peak_program_memory_of_the_smallest_model() {
    timeout(
        JOURNEY_TIMEOUT,
        measure_program_memory("Qwen3.5-0.8B-MLX-4bit"),
    )
    .await
    .expect("the journey must finish inside its bound");
}

#[tokio::test]
#[ignore = "loads a real 2B model and its Neural Engine programs"]
#[serial]
async fn should_measure_peak_program_memory_of_the_two_billion_parameter_model() {
    timeout(JOURNEY_TIMEOUT, measure_program_memory("Qwen3.5-2B-4bit"))
        .await
        .expect("the journey must finish inside its bound");
}
