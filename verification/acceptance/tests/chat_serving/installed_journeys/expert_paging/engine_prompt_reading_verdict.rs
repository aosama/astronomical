//! Which text engine reads a prompt faster, with a profiler verdict
//! (issue #2585).
//!
//! Hand-timed pairs could not answer this: whichever engine ran second in a
//! sitting read a few percent slower as the Mac warmed. Here both engines
//! hold the test run's model in one process and each reads the same prompt
//! as a procedure of the MLX performance profiler: a warm-up, an observed
//! run and twelve interleaved rounds with the order flipped every round.
//! The verdict is on a read's duration.
//!
//! Both engines share one MLX in this process, so `paging_experiment`'s
//! limit must hold the model twice with room to spare; neither engine is
//! then short of memory, which is the case being compared. An engine
//! evaluates on a thread of its own, where the profiler does not yet observe
//! command buffers or blocking waits, so only the duration verdict is
//! printed unless it did observe them.
//!
//! `cargo measure-engine-prompt-reading-verdict --configuration PATH`.
//! Asserts nothing; prints the verdicts and writes the comparison page.

use std::path::Path;

use ipc_protocol::RequestId;
use mlx_performance_profiler::{
    MeasuredProcedure, MetricComparison, ReportLocation, Verdict, compare_procedures,
};
use model_serving::{InferenceEngine, Qwen3_5StreamingEngineTestControls, TextInferenceRequest};
use serial_test::serial;

use super::reply_speed_cases::{essay_prompt_token_ids, read_a_prompt_once};
use crate::serving_acceptance::support::performance_attribution::{
    load_configured_resident_engine, load_paging_experiment_engine,
};

/// Two full resident chunks: long enough that reading is most of a run, and
/// short enough that fourteen runs of each engine take minutes on a dense
/// 27B.
const PROMPT_TOKEN_COUNT: usize = 4_096;
const FIRST_REQUEST_ID: u64 = 98_600;
const REPORT_SUBDIRECTORY: &str = "engine-prompt-reading-verdict";

/// One engine reading the prompt, a new request every run.
struct PromptRead<'journey, Engine> {
    name: &'static str,
    tokio_runtime: &'journey tokio::runtime::Runtime,
    engine: &'journey mut Engine,
    prompt_token_ids: &'journey [u32],
    end_of_sequence_token_ids: &'journey [u32],
    next_request_id: u64,
}

impl<Engine> MeasuredProcedure for PromptRead<'_, Engine>
where
    Engine: InferenceEngine<Request = TextInferenceRequest>,
{
    fn name(&self) -> &str {
        self.name
    }

    /// Nothing to restore: an engine keeps no prompt once its request ends,
    /// and no `astro_ssd_prompt_cache` store is attached.
    fn prepare(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        let request_id = RequestId::new(self.next_request_id);
        self.next_request_id += 1;
        let written_token_ids = self.tokio_runtime.block_on(read_a_prompt_once(
            self.engine,
            (self.prompt_token_ids, self.end_of_sequence_token_ids),
            request_id,
            self.name,
        ));
        if written_token_ids.is_empty() {
            return Err(format!(
                "{}: nothing was written after the prompt",
                self.name
            ));
        }
        Ok(())
    }
}

#[test]
#[ignore = "holds the model on both text engines at once; run through cargo measure-engine-prompt-reading-verdict"]
#[serial]
fn should_give_a_verdict_on_prompt_reading_between_the_two_text_engines() {
    let tokio_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime is built");
    let _direct_mlx_guard = tokio_runtime.block_on(crate::common::direct_mlx_test_guard());
    let mut resident = tokio_runtime.block_on(load_configured_resident_engine(
        "prompt_verdict_resident_model_load",
    ));
    let mut streaming = tokio_runtime.block_on(load_paging_experiment_engine(
        "prompt_verdict_streaming_model_load",
    ));
    // A hosted drafter stays loaded and proposes nothing, so the few tokens
    // after the prompt are plain ones on both engines.
    tokio_runtime
        .block_on(
            streaming
                .engine
                .set_hosted_drafter_writes_replies_for_tests(false),
        )
        .expect("the engine takes the drafter's part");
    let prompt_token_ids = essay_prompt_token_ids(&resident.model_directory, PROMPT_TOKEN_COUNT);
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    let mut resident_read = PromptRead {
        name: "resident_engine",
        tokio_runtime: &tokio_runtime,
        engine: &mut resident.engine,
        prompt_token_ids: &prompt_token_ids,
        end_of_sequence_token_ids: &resident.end_of_sequence_token_ids,
        next_request_id: FIRST_REQUEST_ID,
    };
    let mut streaming_read = PromptRead {
        name: "streaming_engine",
        tokio_runtime: &tokio_runtime,
        engine: &mut streaming.engine,
        prompt_token_ids: &prompt_token_ids,
        end_of_sequence_token_ids: &streaming.end_of_sequence_token_ids,
        next_request_id: FIRST_REQUEST_ID + 1_000,
    };
    let comparison = compare_procedures(
        &mut resident_read,
        &mut streaming_read,
        ReportLocation {
            title: "Prompt reading on the two text engines",
            directory: &report_directory,
            file_base: "prompt_reading",
        },
    )
    .expect("the two engines are compared");
    eprintln!(
        "[engine-prompt-reading] duration: {}",
        streaming_against_resident(&comparison.duration)
    );
    // An engine evaluates on a thread of its own, and the profiler observes
    // command buffers on the calling thread: it sees none, and its GPU busy
    // and idle figures would be stated as measurements that were not taken.
    let has_observed_the_gpu = comparison.baseline.median_command_buffer_count > 0.0
        && comparison.candidate.median_command_buffer_count > 0.0;
    if has_observed_the_gpu {
        eprintln!(
            "[engine-prompt-reading] gpu_busy: {}",
            streaming_against_resident(&comparison.gpu_busy)
        );
        eprintln!(
            "[engine-prompt-reading] gpu_idle: {}",
            streaming_against_resident(&comparison.gpu_idle)
        );
    } else {
        eprintln!(
            "[engine-prompt-reading] gpu_busy and gpu_idle: not observed on the engines' own threads"
        );
    }
    eprintln!(
        "[engine-prompt-reading] page={}",
        report_directory.join("prompt_reading.html").display()
    );
}

/// "streaming faster -3.1% (2.61 against 2.69 s, faster in 11 of 12 rounds)".
fn streaming_against_resident(metric: &MetricComparison) -> String {
    let verdict_text = match metric.verdict {
        Verdict::CandidateFaster => "streaming faster",
        Verdict::WithinSpread => "within spread",
        Verdict::CandidateSlower => "streaming slower",
    };
    format!(
        "{verdict_text} {:+.1}% ({:.3} against {:.3} s, streaming faster in {} of {} rounds)",
        metric.relative_difference * 100.0,
        metric.candidate_median_microseconds / 1e6,
        metric.baseline_median_microseconds / 1e6,
        metric.rounds_candidate_faster,
        metric.round_count
    )
}
