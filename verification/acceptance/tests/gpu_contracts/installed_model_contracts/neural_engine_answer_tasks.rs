//! Do real answers change when work moves to the Neural Engine (issue #161)?
//!
//! Logit cosine rates every Neural Engine arm about the same, and greedy prefixes
//! diverge from any rounding at all, so neither says whether an answer got worse.
//! These tasks hide facts across a long Shakespeare prompt and check the answer
//! exactly: retrieval, ordering, an exact format, and code. Facts sit in chunks
//! the Neural Engine computes, so an error that reaches them has to survive the
//! recurrent state to the end of the prompt to be seen here.

use std::time::Instant;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::NeuralEnginePrefillStatus;
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5ResidentModel, Qwen3_5Tokenizer,
};
use serial_test::serial;
use tokio::time::timeout;

use super::neural_engine_mlp_prefill::{
    CHUNK_TOKEN_COUNT, DENSE_MODEL_ID, FULL_ATTENTION_KV_GROWTH_TOKENS, JOURNEY_TIMEOUT,
    MAXIMUM_OUTPUT_TOKENS, ROMEO_AND_JULIET_SOURCE, greedy_continuation, last_token_logits,
    load_dense_model,
};

const JOURNEY_NAME: &str = "neural-engine-answer-tasks";
// Enough Shakespeare for five full chunks of about 2,048 tokens before the question.
const SHAKESPEARE_WORD_COUNT: usize = 9_000;
// Room for an answer that explains itself before its code, as some arms do.
const ANSWER_TOKEN_COUNT: usize = 256;
const END_OF_TURN: &str = "<|im_end|>";
const GPU_ARM_LABEL: &str = "gpu";

struct AnswerTask {
    name: &'static str,
    /// Placed evenly through the prompt, from early to late.
    notes: &'static [&'static str],
    question: &'static str,
    /// Every one must appear in the answer, in this order.
    expected_in_order: &'static [&'static str],
}

const ANSWER_TASKS: [AnswerTask; 6] = [
    AnswerTask {
        name: "retrieval",
        notes: &[
            "Archivist's note: the vault code for the east gate is 7319.",
            "Archivist's note: the vault code for the west gate is 4826.",
            "Archivist's note: the vault code for the north gate is 5502.",
        ],
        question: "What is the vault code for the west gate? Answer with the number only.",
        expected_in_order: &["4826"],
    },
    AnswerTask {
        name: "retrieval-among-similar",
        notes: &[
            "Archivist's note: the captain of the Osprey is Hale.",
            "Archivist's note: the captain of the Albatross is Reyes.",
            "Archivist's note: the captain of the Kestrel is Okafor.",
            "Archivist's note: the captain of the Heron is Lindqvist.",
        ],
        question: "Who is the captain of the Kestrel? Answer with the name only.",
        expected_in_order: &["Okafor"],
    },
    AnswerTask {
        name: "ordering",
        notes: &[
            "Archivist's note: visitor number 1 to the library was Mara.",
            "Archivist's note: visitor number 3 to the library was Ilya.",
            "Archivist's note: visitor number 2 to the library was Tobias.",
            "Archivist's note: visitor number 4 to the library was Wen.",
        ],
        question: "List the library's visitors by ascending visitor number, separated by commas, \
                   with nothing else.",
        expected_in_order: &["Mara", "Tobias", "Ilya", "Wen"],
    },
    AnswerTask {
        name: "exact-format",
        notes: &[
            "Archivist's note: the ship Osprey carried tea.",
            "Archivist's note: the ship Albatross carried salt.",
            "Archivist's note: the ship Kestrel carried copper.",
        ],
        question: "Write one line per ship in the form NAME=CARGO, in the order the notes gave \
                   them, with nothing else.",
        expected_in_order: &["Osprey=tea", "Albatross=salt", "Kestrel=copper"],
    },
    AnswerTask {
        name: "code-tax",
        notes: &[
            "Archivist's note: the harbour tax rate is 8.25 percent.",
            "Archivist's note: the harbour tax applies only to the part of an amount above 500.",
        ],
        question: "Write a Python function named harbour_tax(amount) that returns the harbour \
                   tax. Output only the code.",
        expected_in_order: &["def harbour_tax", "500", "0.0825"],
    },
    AnswerTask {
        name: "code-cost",
        notes: &[
            "Archivist's note: lantern oil costs 3 coins per hour.",
            "Archivist's note: the first 2 hours of lantern oil are free.",
        ],
        question: "Write a Python function named lantern_cost(hours) that returns the cost in \
                   coins. Output only the code.",
        expected_in_order: &["def lantern_cost", "2", "3"],
    },
];

#[derive(Clone, Copy)]
struct AnswerArm {
    label: &'static str,
    is_mlp_on_neural_engine: bool,
    is_output_gate_on_neural_engine: bool,
}

const ANSWER_ARMS: [AnswerArm; 4] = [
    AnswerArm {
        label: GPU_ARM_LABEL,
        is_mlp_on_neural_engine: false,
        is_output_gate_on_neural_engine: false,
    },
    AnswerArm {
        label: "output-gate",
        is_mlp_on_neural_engine: false,
        is_output_gate_on_neural_engine: true,
    },
    AnswerArm {
        label: "mlp",
        is_mlp_on_neural_engine: true,
        is_output_gate_on_neural_engine: false,
    },
    AnswerArm {
        label: "mlp+output-gate",
        is_mlp_on_neural_engine: true,
        is_output_gate_on_neural_engine: true,
    },
];

fn rendered_task_prompt(task: &AnswerTask) -> String {
    let words: Vec<&str> = ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(SHAKESPEARE_WORD_COUNT)
        .collect();
    let words_between_notes = words.len() / (task.notes.len() + 1);
    let mut document = String::new();
    for (word_index, word) in words.iter().enumerate() {
        if word_index > 0 && word_index % words_between_notes == 0 {
            let note_index = word_index / words_between_notes - 1;
            if let Some(note) = task.notes.get(note_index) {
                document.push_str("\n\n");
                document.push_str(note);
                document.push_str("\n\n");
            }
        }
        document.push_str(word);
        document.push(' ');
    }
    format!(
        "<|im_start|>user\n{document}\n\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        task.question
    )
}

/// Greedy decoding runs a fixed length, so what follows the end of the turn is
/// not part of the answer.
fn decoded_answer(tokenizer: &Qwen3_5Tokenizer, answer_token_ids: &[u32]) -> String {
    let end_of_turn_token_ids = tokenizer
        .encode_prompt(END_OF_TURN)
        .expect("the end of turn encodes");
    let mut decoder = tokenizer.incremental_decoder();
    let mut answer = String::new();
    for token_id in answer_token_ids {
        if end_of_turn_token_ids.contains(token_id) {
            break;
        }
        if let Some(decoded_text) = decoder.push_token(*token_id).expect("the token decodes") {
            answer.push_str(&decoded_text);
        }
    }
    answer.trim().to_owned()
}

fn contains_in_order(answer: &str, expected_in_order: &[&str]) -> bool {
    let mut remaining_answer = answer;
    for expected in expected_in_order {
        match remaining_answer.split_once(expected) {
            Some((_, after_expected)) => remaining_answer = after_expected,
            None => return false,
        }
    }
    true
}

/// Full chunks take the prefill path the programs serve, and the tail with the
/// question stays on the GPU, as it does when serving.
fn answer_token_ids(model: &Qwen3_5ResidentModel, prompt_token_ids: &[u32]) -> Vec<u32> {
    let mut decoder_state = model
        .empty_request_decoder_state_matching_loaded_weights(FULL_ATTENTION_KV_GROWTH_TOKENS)
        .expect("the state allocates");
    let full_chunk_count = (prompt_token_ids.len() - 1) / CHUNK_TOKEN_COUNT;
    for chunk_index in 0..full_chunk_count {
        let chunk_start = chunk_index * CHUNK_TOKEN_COUNT;
        model
            .prefill_chunk(
                &prompt_token_ids[chunk_start..chunk_start + CHUNK_TOKEN_COUNT],
                chunk_start as u32,
                &mut decoder_state,
            )
            .expect("the full chunk runs");
    }
    let tail_start = full_chunk_count * CHUNK_TOKEN_COUNT;
    let tail_logits = model
        .forward_chunk(
            &prompt_token_ids[tail_start..],
            tail_start as u32,
            &mut decoder_state,
        )
        .expect("the tail runs");
    tail_logits.evaluate().expect("the tail evaluates");
    greedy_continuation(
        model,
        &last_token_logits(model.runtime(), &tail_logits),
        &mut decoder_state,
        prompt_token_ids.len(),
        ANSWER_TOKEN_COUNT,
    )
}

struct TaskRun<'run> {
    model: &'run Qwen3_5ResidentModel,
    tokenizer: &'run Qwen3_5Tokenizer,
    started_at: Instant,
    finished_count: usize,
    total_count: usize,
}

impl TaskRun<'_> {
    fn report_progress(&mut self) {
        self.finished_count += 1;
        let elapsed_seconds = self.started_at.elapsed().as_secs_f64();
        eprintln!(
            "[{JOURNEY_NAME}] status=progress {}/{} ({:.0}%) elapsed_s={elapsed_seconds:.0} eta_s={:.0}",
            self.finished_count,
            self.total_count,
            100.0 * self.finished_count as f64 / self.total_count as f64,
            elapsed_seconds / self.finished_count as f64
                * (self.total_count - self.finished_count) as f64,
        );
    }

    /// Every arm answers the same prompt on one model, so a difference between
    /// answers can only come from which engine computed what.
    fn run_task(&mut self, task: &AnswerTask) {
        let prompt_token_ids = self
            .tokenizer
            .encode_prompt(&rendered_task_prompt(task))
            .expect("the task prompt encodes");
        let mut gpu_answer = String::new();
        for arm in ANSWER_ARMS {
            self.model
                .set_neural_engine_mlp_prefill_paused(!arm.is_mlp_on_neural_engine);
            self.model
                .set_neural_engine_output_gate_paused(!arm.is_output_gate_on_neural_engine);
            let token_ids = answer_token_ids(self.model, &prompt_token_ids);
            let answer = decoded_answer(self.tokenizer, &token_ids);
            if arm.label == GPU_ARM_LABEL {
                gpu_answer.clone_from(&answer);
            }
            eprintln!(
                "[{JOURNEY_NAME}] MEASURED task={} arm={} passed={} same_answer_as_gpu={} answer={answer:?}",
                task.name,
                arm.label,
                contains_in_order(&answer, task.expected_in_order),
                answer == gpu_answer,
            );
            self.report_progress();
        }
    }
}

fn load_tokenizer(development_config: &AstronomicalConfig) -> Qwen3_5Tokenizer {
    let discovered_model =
        crate::common::configured_discovered_model_by_id(development_config, DENSE_MODEL_ID);
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&discovered_model.model_directory, MAXIMUM_OUTPUT_TOKENS)
        .expect("the artifact passes production validation");
    Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact).expect("the tokenizer loads")
}

fn prepare_both_program_families(model: &mut Qwen3_5ResidentModel, maximum_context_tokens: u32) {
    let mut chunking = crate::common::standard_worker_chunking_configuration();
    chunking.experimental_neural_engine_prefill_enabled = true;
    let status = model.prepare_neural_engine_prefill_matching_production(
        &chunking,
        maximum_context_tokens,
        None,
    );
    assert!(matches!(status, NeuralEnginePrefillStatus::Active { .. }));
}

async fn compare_answers_across_arms() {
    let _gpu_exclusive = crate::common::direct_mlx_test_guard().await;
    let development_config =
        AstronomicalConfig::load_from_development_location().expect("the config loads");
    let tokenizer = load_tokenizer(&development_config);
    let (mut model, _, maximum_context_tokens) = load_dense_model(&development_config);
    prepare_both_program_families(&mut model, maximum_context_tokens);
    let mut task_run = TaskRun {
        model: &model,
        tokenizer: &tokenizer,
        started_at: Instant::now(),
        finished_count: 0,
        total_count: ANSWER_TASKS.len() * ANSWER_ARMS.len(),
    };
    for task in &ANSWER_TASKS {
        task_run.run_task(task);
    }
}

#[tokio::test]
#[ignore = "loads a real dense artifact and both program families, then answers four tasks per arm"]
#[serial]
async fn should_compare_real_answers_across_neural_engine_arms() {
    timeout(JOURNEY_TIMEOUT, compare_answers_across_arms())
        .await
        .expect("the answer task journey must finish inside its bound");
}

#[test]
fn should_require_every_expected_part_in_order() {
    assert!(contains_in_order(
        "Mara, Tobias, Ilya, Wen",
        &["Mara", "Tobias", "Ilya", "Wen"]
    ));
}

#[test]
fn should_reject_expected_parts_out_of_order() {
    assert!(!contains_in_order(
        "Mara, Ilya, Tobias, Wen",
        &["Mara", "Tobias", "Ilya", "Wen"]
    ));
}

#[test]
fn should_reject_a_missing_expected_part_and_an_empty_answer() {
    assert!(!contains_in_order("4827", &["4826"]));
    assert!(!contains_in_order("", &["4826"]));
}

#[test]
fn should_place_every_note_inside_the_prompt_for_each_task() {
    for task in &ANSWER_TASKS {
        let prompt = rendered_task_prompt(task);
        for note in task.notes {
            assert!(prompt.contains(note), "{} lost a note", task.name);
        }
    }
}
