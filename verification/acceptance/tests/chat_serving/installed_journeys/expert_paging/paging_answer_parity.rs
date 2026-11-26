//! The headline claim: reading experts from SSD must not change the answer.
//!
//! The same model and Shakespeare prompts run on the text engine twice: under a limit
//! that holds the whole model, and under ceilings that force paging. Under a ceiling the
//! engine serves experts from pages and evicts tables it would otherwise keep, so any
//! way that changes the arithmetic shows up here. Each question is asked twice, because
//! a second ask reuses whatever the first left behind.
//!
//! Three questions, because each catches what the others cannot:
//! - a long prompt answered greedily, where the top token has to survive paging;
//! - a prompt of three spans, read one layer at a time, whose first paged ask has one
//!   span fail the way MLX reports a span it could not fit. The engine must run that
//!   span again, once, keep most of what it had in memory, and give the same answer
//!   (issue #2221);
//! - a short prompt answered with a fixed seed. A seeded draw flips on logit changes far
//!   smaller than greedy ever sees, and a short prompt is one prefill chunk small enough
//!   that MLX picks its quantized expert kernel by the size of the page it is handed
//!   unless the declared assignment order prevents it (issue #1679).
//!
//! MLX fixes its memory limit once per process, so the whole-model answers come from a
//! child process running the recording test below, and the paged engine runs in this
//! one. That also leaves the two runs with no process state in common. The child prints
//! its answers on marked lines rather than writing files, so an answer can never come
//! from an earlier run.

use std::{path::Path, process::Stdio, time::Duration};

use ipc_protocol::{ExpertMemoryMode, RequestId};
use model_serving::{
    CompleteResidencyHeadroomBoundary, Qwen3_5ArtifactValidator, Qwen3_5TextEngine,
    Qwen3_5TextEngineTestControls, TextModelTestControls,
    model_memory_geometry_from_validated_artifact,
};
use runtime_integration::MlxMemoryLimits;
use serial_test::serial;
use tokio::{process::Command, time::timeout};

use super::failed_span_redo::{
    RedoCost, arm_span_failure, assert_one_span_was_redone_cheaply, redo_cost_since,
};
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, counter_amount, create_attributed_engine, generation_report_for_request,
    load_engine_with_progress, read_attribution_report_documents, run_generation_with_sampling,
};

/// One chunk holds the long prompt whole, so both runs prefill it in one pass.
const PREFILL_CHUNK_TOKEN_COUNT: usize = 1_024;
const OUTPUT_TOKEN_COUNT: u16 = 64;
const JOURNEY_BUDGET: Duration = Duration::from_secs(120);
const RECORDING_TEST_NAME: &str = "should_record_the_whole_model_answer_for_the_parity_journeys";
/// Starts each line on which the recording child reports one answer.
const WHOLE_MODEL_ANSWER_MARKER: &str = "[parity-whole-model-answer]";

#[derive(Clone, Copy)]
struct ParityQuestion {
    name: &'static str,
    prompt_token_count: usize,
    generation_sampling: GenerationSampling,
    first_request_id: RequestId,
    second_request_id: RequestId,
    /// The span, as (layer, span), the first paged ask fails once on purpose.
    span_to_fail_once: Option<(usize, usize)>,
}

const LONG_GREEDY_QUESTION: ParityQuestion = ParityQuestion {
    name: "long_greedy",
    prompt_token_count: 1_024,
    generation_sampling: GenerationSampling::Greedy,
    first_request_id: RequestId::new(96_200),
    second_request_id: RequestId::new(96_201),
    span_to_fail_once: None,
};

/// Three spans of the prefill chunk. The second span of the third layer fails, which
/// puts that layer back to its first span because that one was not confirmed yet.
const REDONE_SPAN_QUESTION: ParityQuestion = ParityQuestion {
    name: "three_span_greedy_with_a_redone_span",
    prompt_token_count: 2_560,
    generation_sampling: GenerationSampling::Greedy,
    first_request_id: RequestId::new(96_204),
    second_request_id: RequestId::new(96_205),
    span_to_fail_once: Some((2, 1)),
};

/// 48 tokens route 384 assignments, under four per expert of a 256-expert layer.
const SHORT_SEEDED_QUESTION: ParityQuestion = ParityQuestion {
    name: "short_seeded",
    prompt_token_count: 48,
    generation_sampling: GenerationSampling::Seeded,
    first_request_id: RequestId::new(96_202),
    second_request_id: RequestId::new(96_203),
    span_to_fail_once: None,
};

const PARITY_QUESTIONS: [ParityQuestion; 3] = [
    LONG_GREEDY_QUESTION,
    SHORT_SEEDED_QUESTION,
    REDONE_SPAN_QUESTION,
];

/// The first and second answer to one question.
#[derive(Debug, Default, Eq, PartialEq)]
struct AnswerPair {
    first_generated_token_ids: Vec<u32>,
    second_generated_token_ids: Vec<u32>,
}

#[derive(Clone, Copy)]
enum PagingDepth {
    /// Every expert fits but the activation headroom does not.
    Boundary,
    /// Half the expert payload is out of RAM at any moment.
    HalfOfExperts,
}

fn paging_ceiling_bytes(model_directory: &Path, paging_depth: PagingDepth) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(OUTPUT_TOKEN_COUNT))
        .expect("the configured sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        model_memory_geometry_from_validated_artifact(&validated_artifact, model_directory)
            .expect("the sparse artifact should expose its RAM geometry");
    let residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    );
    let boundary_ceiling_bytes = residency_boundary
        .paging_ceiling_bytes()
        .expect("a sparse artifact has a ceiling that forces paging");
    match paging_depth {
        PagingDepth::Boundary => boundary_ceiling_bytes,
        PagingDepth::HalfOfExperts => boundary_ceiling_bytes
            .checked_sub(model_geometry.complete_expert_payload_bytes / 2)
            .expect("half the expert payload should fit under the boundary ceiling"),
    }
}

async fn ask_twice(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    question: ParityQuestion,
    prompt_token_ids: &[u32],
    end_of_sequence_token_ids: &[u32],
) -> AnswerPair {
    let phase_name = format!("parity_{}", question.name);
    let first_generated_token_ids = run_generation_with_sampling(
        qwen3_5_engine,
        question.first_request_id,
        prompt_token_ids,
        &phase_name,
        OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
        question.generation_sampling,
    )
    .await;
    let second_generated_token_ids = run_generation_with_sampling(
        qwen3_5_engine,
        question.second_request_id,
        prompt_token_ids,
        &phase_name,
        OUTPUT_TOKEN_COUNT,
        end_of_sequence_token_ids,
        question.generation_sampling,
    )
    .await;
    AnswerPair {
        first_generated_token_ids,
        second_generated_token_ids,
    }
}

/// Asks one question twice. When `fails_spans` and the question names a span to fail,
/// that span fails on its first ask, and what running it again cost is returned with
/// the answers.
async fn ask_twice_counting_redos(
    qwen3_5_engine: &mut Qwen3_5TextEngine,
    (question, fails_spans): (ParityQuestion, bool),
    prompt_token_ids: &[u32],
    end_of_sequence_token_ids: &[u32],
) -> (AnswerPair, Option<RedoCost>) {
    let span_to_fail_once = question.span_to_fail_once.filter(|_| fails_spans);
    let Some(span_to_fail_once) = span_to_fail_once else {
        let answer_pair = ask_twice(
            qwen3_5_engine,
            question,
            prompt_token_ids,
            end_of_sequence_token_ids,
        )
        .await;
        return (answer_pair, None);
    };
    let redo_baseline = arm_span_failure(qwen3_5_engine, span_to_fail_once).await;
    let answer_pair = ask_twice(
        qwen3_5_engine,
        question,
        prompt_token_ids,
        end_of_sequence_token_ids,
    )
    .await;
    let redo_cost = redo_cost_since(qwen3_5_engine, redo_baseline).await;
    (answer_pair, Some(redo_cost))
}

struct ParityRun {
    answer_pairs: Vec<AnswerPair>,
    /// What the question that fails a span cost.
    redo_cost: Option<RedoCost>,
    first_expert_bytes_read: u64,
    expert_memory_mode: Option<ExpertMemoryMode>,
}

/// Asks every question twice on one engine under `mlx_memory_limits`, in order, so later
/// questions meet the tables earlier ones left.
async fn ask_every_question(
    model_directory: &Path,
    mlx_memory_limits: &MlxMemoryLimits,
    fails_spans: bool,
) -> ParityRun {
    let attribution_directory =
        tempfile::tempdir().expect("the journey should create an attribution directory");
    let attribution_log_path = attribution_directory.path().join("parity.jsonl");
    let (mut qwen3_5_engine, end_of_sequence_token_ids) = create_attributed_engine(
        model_directory,
        &attribution_log_path,
        mlx_memory_limits,
        PREFILL_CHUNK_TOKEN_COUNT as u32,
    );
    load_engine_with_progress(&mut qwen3_5_engine, "parity").await;
    // A package may carry a drafter. Drafted rows are verified by a batched
    // kernel whose logits differ from the one-row step in the last bits, so a
    // drafted reply depends on which drafts were kept, not only on paging.
    // The drafter stays hosted and silent, so this journey judges paging alone.
    qwen3_5_engine
        .set_hosted_drafter_writes_replies_for_tests(false)
        .await
        .expect("the engine takes the drafter's part");
    let expert_memory_mode = qwen3_5_engine
        .expert_memory_mode_for_tests()
        .await
        .expect("the loaded model should expose its expert mode");
    let mut answer_pairs = Vec::with_capacity(PARITY_QUESTIONS.len());
    let mut redo_cost = None;
    for question in PARITY_QUESTIONS {
        let (answer_pair, question_redo_cost) = ask_twice_counting_redos(
            &mut qwen3_5_engine,
            (question, fails_spans),
            &parity_prompt_token_ids(model_directory, question),
            &end_of_sequence_token_ids,
        )
        .await;
        answer_pairs.push(answer_pair);
        redo_cost = redo_cost.or(question_redo_cost);
    }
    let attribution_documents = read_attribution_report_documents(&attribution_log_path);
    let first_report = generation_report_for_request(
        &attribution_documents,
        LONG_GREEDY_QUESTION.first_request_id.value(),
    );
    ParityRun {
        answer_pairs,
        redo_cost,
        first_expert_bytes_read: counter_amount(first_report, "positional_file_read_byte_count"),
        expert_memory_mode,
    }
}

fn first_divergence(reference_token_ids: &[u32], compared_token_ids: &[u32]) -> Option<usize> {
    for (token_position, reference_token_id) in reference_token_ids.iter().enumerate() {
        if compared_token_ids.get(token_position) != Some(reference_token_id) {
            return Some(token_position);
        }
    }
    (compared_token_ids.len() != reference_token_ids.len()).then_some(reference_token_ids.len())
}

fn assert_same_answer(reference_token_ids: &[u32], compared_token_ids: &[u32], run_name: &str) {
    assert_eq!(
        first_divergence(reference_token_ids, compared_token_ids),
        None,
        "{run_name} changed the answer: whole model {reference_token_ids:?}, {run_name} {compared_token_ids:?}"
    );
}

fn parity_prompt_token_ids(model_directory: &Path, question: ParityQuestion) -> Vec<u32> {
    crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
        model_directory,
        crate::common::resident_sparse_moe_model_id(),
        question.first_request_id,
        question.prompt_token_count,
        OUTPUT_TOKEN_COUNT,
    )
}

fn parity_model_directory() -> std::path::PathBuf {
    crate::common::configured_installed_model_directory_by_id(
        crate::common::resident_sparse_moe_model_id(),
    )
}

/// One marked line: the question, which ask, then the token IDs.
fn whole_model_answer_line(
    question_name: &str,
    ask_name: &str,
    generated_token_ids: &[u32],
) -> String {
    let mut answer_line = format!("{WHOLE_MODEL_ANSWER_MARKER} {question_name} {ask_name}");
    for generated_token_id in generated_token_ids {
        answer_line.push(' ');
        answer_line.push_str(&generated_token_id.to_string());
    }
    answer_line
}

/// Collects one question's two answers from the child's output. `None` when either
/// answer is missing, so a child that ran no test cannot pass as one that answered.
fn parsed_answer_pair(child_output: &str, question_name: &str) -> Option<AnswerPair> {
    let mut first_generated_token_ids = None;
    let mut second_generated_token_ids = None;
    for output_line in child_output.lines() {
        let mut line_words = output_line.split_whitespace();
        if line_words.next() != Some(WHOLE_MODEL_ANSWER_MARKER)
            || line_words.next() != Some(question_name)
        {
            continue;
        }
        let ask_name = line_words.next();
        let mut generated_token_ids = Vec::new();
        for token_text in line_words {
            generated_token_ids.push(token_text.parse().ok()?);
        }
        match ask_name {
            Some("first") => first_generated_token_ids = Some(generated_token_ids),
            Some("second") => second_generated_token_ids = Some(generated_token_ids),
            _ => return None,
        }
    }
    Some(AnswerPair {
        first_generated_token_ids: first_generated_token_ids?,
        second_generated_token_ids: second_generated_token_ids?,
    })
}

/// Runs the recording test in its own process and returns the whole-model answer
/// pairs, in `PARITY_QUESTIONS` order. The child is awaited, not blocked on, so the
/// journey budget covers it, and it is killed if the budget runs out.
async fn whole_model_answers_from_child_process() -> Vec<AnswerPair> {
    let test_executable = std::env::current_exe().expect("the test should know its executable");
    let (_, module_inside_target) = module_path!()
        .split_once("::")
        .expect("this test sits inside a test target");
    let recording_test = format!("{module_inside_target}::{RECORDING_TEST_NAME}");
    let recording_output = Command::new(test_executable)
        .args([
            recording_test.as_str(),
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .output()
        .await
        .expect("the whole-model recording process should start");
    assert!(
        recording_output.status.success(),
        "the whole-model recording process failed"
    );
    let child_output = String::from_utf8_lossy(&recording_output.stdout);
    let mut whole_model_answer_pairs = Vec::with_capacity(PARITY_QUESTIONS.len());
    for question in PARITY_QUESTIONS {
        match parsed_answer_pair(&child_output, question.name) {
            Some(whole_model_answer_pair) => whole_model_answer_pairs.push(whole_model_answer_pair),
            None => panic!(
                "the whole-model recording process gave no answers to {}",
                question.name
            ),
        }
    }
    whole_model_answer_pairs
}

fn assert_question_parity(
    question: ParityQuestion,
    whole_model_answer_pair: &AnswerPair,
    paged_answer_pair: &AnswerPair,
) {
    let whole_model_answer = &whole_model_answer_pair.first_generated_token_ids;
    assert_same_answer(
        whole_model_answer,
        &whole_model_answer_pair.second_generated_token_ids,
        &format!("the second whole-model {} ask", question.name),
    );
    assert_same_answer(
        whole_model_answer,
        &paged_answer_pair.first_generated_token_ids,
        &format!("the first paged {} ask", question.name),
    );
    assert_same_answer(
        whole_model_answer,
        &paged_answer_pair.second_generated_token_ids,
        &format!("the second paged {} ask", question.name),
    );
}

async fn prove_paging_keeps_the_answer(paging_depth: PagingDepth, depth_name: &str) {
    let whole_model_answer_pairs = whole_model_answers_from_child_process().await;
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = parity_model_directory();
    let ceiling_limit_bytes = usize::try_from(paging_ceiling_bytes(&model_directory, paging_depth))
        .expect("the ceiling fits usize");
    let ceiling_limits = MlxMemoryLimits::new(ceiling_limit_bytes, ceiling_limit_bytes)
        .expect("the ceiling should be a valid MLX limit");
    let paged_run = ask_every_question(&model_directory, &ceiling_limits, true).await;
    assert_ne!(
        paged_run.expert_memory_mode,
        Some(ExpertMemoryMode::Resident),
        "this run must not seat every expert"
    );
    assert!(
        paged_run.first_expert_bytes_read > 0,
        "the paged run must really read experts from disk"
    );
    for ((question, whole_model_answer_pair), paged_answer_pair) in PARITY_QUESTIONS
        .into_iter()
        .zip(&whole_model_answer_pairs)
        .zip(&paged_run.answer_pairs)
    {
        eprintln!(
            "[paging-answer-parity] status=progress depth={depth_name} question={} paged_expert_bytes_read={} whole_model_tokens={:?} paged_tokens={:?}",
            question.name,
            paged_run.first_expert_bytes_read,
            whole_model_answer_pair.first_generated_token_ids,
            paged_answer_pair.first_generated_token_ids,
        );
        assert_question_parity(question, whole_model_answer_pair, paged_answer_pair);
    }
    let redo_cost = paged_run
        .redo_cost
        .as_ref()
        .expect("one question fails a span on purpose");
    assert_one_span_was_redone_cheaply(redo_cost, depth_name);
    eprintln!(
        "[paging-answer-parity] status=success depth={depth_name} redone_spans={} kept_expert_gb_before={:.2} kept_expert_gb_after={:.2}",
        redo_cost.redo_attempt_count,
        redo_cost.kept_expert_bytes_before as f64 / 1_073_741_824.0,
        redo_cost.kept_expert_bytes_after as f64 / 1_073_741_824.0,
    );
}

/// Not a journey of its own: the parity journeys run this in a child process and read
/// its answers from standard output. A sweep of ignored tests also runs it alone, which
/// costs one load under a limit that holds the whole model.
#[tokio::test]
#[ignore = "loads a sparse model whole; run by the parity journeys in a child process"]
#[serial]
async fn should_record_the_whole_model_answer_for_the_parity_journeys() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = parity_model_directory();
    let mlx_memory_limits =
        crate::common::sample_machine_serving_acceptance_mlx_memory_limits().await;
    assert!(
        mlx_memory_limits.active_memory_limit_bytes() as u64
            > paging_ceiling_bytes(&model_directory, PagingDepth::Boundary),
        "the reference run's limit must hold the whole model"
    );
    let whole_model_run = ask_every_question(&model_directory, &mlx_memory_limits, false).await;
    for (question, answer_pair) in PARITY_QUESTIONS
        .into_iter()
        .zip(&whole_model_run.answer_pairs)
    {
        for (ask_name, generated_token_ids) in [
            ("first", &answer_pair.first_generated_token_ids),
            ("second", &answer_pair.second_generated_token_ids),
        ] {
            println!(
                "{}",
                whole_model_answer_line(question.name, ask_name, generated_token_ids)
            );
        }
    }
}

#[tokio::test]
#[ignore = "loads a sparse model whole and paged, and compares the answers"]
#[serial]
async fn should_give_the_whole_model_answer_when_paging_at_the_residency_boundary() {
    timeout(
        JOURNEY_BUDGET,
        prove_paging_keeps_the_answer(PagingDepth::Boundary, "boundary"),
    )
    .await
    .expect("the parity journey should finish within its budget");
}

#[tokio::test]
#[ignore = "loads a sparse model whole and paged, and compares the answers"]
#[serial]
async fn should_give_the_whole_model_answer_when_half_the_experts_are_out_of_ram() {
    timeout(
        JOURNEY_BUDGET,
        prove_paging_keeps_the_answer(PagingDepth::HalfOfExperts, "half_of_experts"),
    )
    .await
    .expect("the parity journey should finish within its budget");
}

#[test]
fn should_read_back_both_answers_the_child_printed() {
    let child_output = format!(
        "running 1 test\n{}\n{}\ntest ok\n",
        whole_model_answer_line("short_seeded", "first", &[7, 0, 248_069]),
        whole_model_answer_line("short_seeded", "second", &[]),
    );
    assert_eq!(
        parsed_answer_pair(&child_output, "short_seeded"),
        Some(AnswerPair {
            first_generated_token_ids: vec![7, 0, 248_069],
            second_generated_token_ids: Vec::new(),
        })
    );
}

#[test]
fn should_find_no_answers_when_the_child_ran_no_test() {
    assert_eq!(
        parsed_answer_pair("running 0 tests\ntest result: ok\n", "long_greedy"),
        None
    );
}

#[test]
fn should_find_no_answers_when_only_one_ask_or_another_question_was_printed() {
    let child_output = format!(
        "{}\n{}\n",
        whole_model_answer_line("long_greedy", "first", &[1, 2]),
        whole_model_answer_line("short_seeded", "second", &[3]),
    );
    assert_eq!(parsed_answer_pair(&child_output, "long_greedy"), None);
    assert_eq!(parsed_answer_pair(&child_output, "short_seeded"), None);
}

#[test]
fn should_reject_an_answer_line_with_a_token_that_is_not_a_number() {
    let child_output = format!(
        "{WHOLE_MODEL_ANSWER_MARKER} long_greedy first 1 two\n{}\n",
        whole_model_answer_line("long_greedy", "second", &[1, 2]),
    );
    assert_eq!(parsed_answer_pair(&child_output, "long_greedy"), None);
}

#[test]
fn should_report_no_divergence_for_identical_answers() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
    assert_eq!(first_divergence(&[], &[]), None);
}

#[test]
fn should_report_the_first_position_that_differs() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
    assert_eq!(first_divergence(&[1, 2, 3], &[9, 9, 9]), Some(0));
}

#[test]
fn should_report_a_shorter_or_longer_answer_as_a_divergence() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2]), Some(2));
    assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
}
