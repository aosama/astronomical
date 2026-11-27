//! Whole-model training learns what the final layers cannot, on a real
//! checkpoint.
//!
//! Each example is a short Romeo and Juliet line opening with a code,
//! `[code 17]` or `[code 42]`, and the code alone decides the right answer;
//! the words say nothing about it. The shipped model cannot know the codes,
//! and the final layers only re-weigh what the rest of the model already
//! reads, so they cannot learn them either (measured through `decision
//! train` on Kev: 53% right before and 43% after). A whole-model training
//! must learn them: this journey trains on 140 lines, chooses its stopping
//! point on 30, and requires the trained model to be right on at least 90%
//! of 30 lines it never saw: the split `decision train` makes of its
//! 200-question minimum.
//!
//! Smaller jobs were tried and do not prove it: the add-ons need about 60
//! optimizer steps before the code shows, and a run stops after three
//! passes without improvement, so with 40 or 64 training lines (3 or 4
//! steps a pass) it stopped before learning anything. At this size it
//! takes about a minute and a half on Kev 0.8B (53% to 100% right) and
//! about fifteen on Clef 9B (50% to 100%), whose 108 steps cost about eight
//! and a half seconds each. Laya needs its own recipe
//! (`LAYA_WHOLE_MODEL_TRAINING_SETTINGS`, issue #2705): on MLX's CUDA backend
//! it takes about four minutes (63% to 100% right), and a second journey
//! repeats it on a balanced layout from another passage.
//!
//! Run one family with its alias and `--configuration PATH` naming that
//! family's `model_directory`:
//! `cargo test-kev-whole-model-positive-control`,
//! `cargo test-clef-whole-model-positive-control` or
//! `cargo test-laya-whole-model-positive-control`.

use decision_contract::OrderedJsonValue;
use decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, QuestionRole, TrainingExample, TrainingScope,
};
use decision_contract::head_tuning_file::TunableHeadFamily;
use test_harness::test_run::TestRunContext;

use super::whole_model_training::{loaded_engine, train_to_completion};

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const EXAMPLE_COUNT: usize = 200;
const WORDS_PER_LINE: usize = 14;
/// The share of judged lines the trained model must answer right.
const REQUIRED_TRAINED_ACCURACY: f64 = 0.9;
const QUESTIONS_JSON: &str = r#"{"mood": {"type": "choice", "instructions": "What is the main feeling of this line from Romeo and Juliet?", "criteria": {"love": "romantic love or longing", "conflict": "quarrels, insults, threats or fighting"}}}"#;

/// How the coded lines are laid out: which option each line's code decides,
/// and which words of the play the lines come from.
#[derive(Clone, Copy)]
struct CodeLayout {
    /// The code of line `i` decides option
    /// `(i * multiplier + i / divisor) % 2`: a fixed scramble of the index,
    /// so neither the line's words nor its position predict it.
    multiplier: usize,
    divisor: usize,
    /// The first word of the first line.
    first_word_index: usize,
}

/// The control's layout. A third of its lines carry option 1, so a guess of
/// option 0 alone is already right about two lines in three.
const DEFAULT_CODE_LAYOUT: CodeLayout = CodeLayout {
    multiplier: 7,
    divisor: 3,
    first_word_index: 0,
};

/// Half the lines carry each option, from another stretch of the play, so
/// nothing the default layout fixed (its lean, its lines) is what is learned
/// (issue #2705).
const BALANCED_CODE_LAYOUT: CodeLayout = CodeLayout {
    multiplier: 5,
    divisor: 4,
    first_word_index: 5_000,
};

impl CodeLayout {
    const fn coded_option(self, example_index: usize) -> usize {
        (example_index * self.multiplier + example_index / self.divisor) % 2
    }
}

fn example_role(example_index: usize, correct_option: usize) -> QuestionRole {
    // Of every twenty lines 14 train, 3 choose and 3 are judged: the split
    // `decision train` makes of its 200-question minimum.
    match example_index % 20 {
        0..14 => QuestionRole::Training { correct_option },
        14..17 => QuestionRole::Choosing { correct_option },
        _ => QuestionRole::Judging { correct_option },
    }
}

pub(super) fn coded_job() -> DecisionTrainingJob {
    coded_job_laid_out(DEFAULT_CODE_LAYOUT)
}

fn coded_job_laid_out(code_layout: CodeLayout) -> DecisionTrainingJob {
    let mut examples = Vec::with_capacity(EXAMPLE_COUNT);
    for example_index in 0..EXAMPLE_COUNT {
        let correct_option = code_layout.coded_option(example_index);
        let code = if correct_option == 0 { 17 } else { 42 };
        let first_word_index = code_layout.first_word_index + example_index * WORDS_PER_LINE;
        let line = romeo_and_juliet::passage(first_word_index, WORDS_PER_LINE);
        examples.push(TrainingExample {
            state: OrderedJsonValue::Text(format!("[code {code}] {line}")),
            questions: serde_json::from_str(QUESTIONS_JSON).expect("the question is well formed"),
            question_roles: vec![example_role(example_index, correct_option)],
        });
    }
    DecisionTrainingJob {
        examples,
        scope: TrainingScope::WholeModel,
    }
}

/// The lines held back from training, which the trained model scores: the
/// ones that pick the stopping point and the unseen ones. Training lines are
/// not among them: the trained model scores only the rows it did not train
/// on (`judged_rows`), so a training line's returned scores are
/// placeholders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeldBackRole {
    Choosing,
    Judging,
}

impl HeldBackRole {
    const fn holds(self, question_role: QuestionRole) -> bool {
        matches!(
            (self, question_role),
            (Self::Choosing, QuestionRole::Choosing { .. })
                | (Self::Judging, QuestionRole::Judging { .. })
        )
    }
}

/// The share of `held_back_role`'s questions whose highest score is the
/// right option, served and trained. Both held-back roles are reported, so
/// a run that stopped early reads differently from one that learned but did
/// not carry the code over.
fn held_back_accuracy(
    job: &DecisionTrainingJob,
    outcome: &DecisionTrainingOutcome,
    held_back_role: HeldBackRole,
) -> (f64, f64) {
    let (mut counted, mut served_right, mut trained_right) = (0, 0, 0);
    for (example, trained_questions) in job.examples.iter().zip(&outcome.examples) {
        let question_role = example.question_roles[0];
        let Some(correct_option) = question_role.correct_option() else {
            continue;
        };
        if !held_back_role.holds(question_role) {
            continue;
        }
        counted += 1;
        if highest_scoring(&trained_questions[0].served.option_logits) == correct_option {
            served_right += 1;
        }
        if highest_scoring(&trained_questions[0].trained_logits) == correct_option {
            trained_right += 1;
        }
    }
    (
        f64::from(served_right) / f64::from(counted),
        f64::from(trained_right) / f64::from(counted),
    )
}

/// Prints the choosing lines' accuracy and every run's summary; the unseen
/// lines' accuracy is printed with the verdict.
fn print_training_report(job: &DecisionTrainingJob, outcome: &DecisionTrainingOutcome) {
    let (served_accuracy, trained_accuracy) =
        held_back_accuracy(job, outcome, HeldBackRole::Choosing);
    eprintln!(
        "[positive-control] choosing_right_served={served_accuracy:.3} choosing_right_trained={trained_accuracy:.3}"
    );
    for run_summary in &outcome.runs {
        eprintln!("[positive-control] run={run_summary:?}");
    }
}

/// Prints the largest trained add-on `up` value. Every `up` starts at zero,
/// so its size after training shows how far training moved the layers at
/// all: a run that never moved reads near zero.
fn print_add_on_movement(outcome: &DecisionTrainingOutcome) {
    let mut largest_up_value = 0.0_f32;
    let mut up_tensor_count = 0;
    for adapter_tensor in &outcome.adapter_tensors {
        if !adapter_tensor.name.ends_with(".adapter_up") {
            continue;
        }
        up_tensor_count += 1;
        for up_value in &adapter_tensor.values.0 {
            largest_up_value = largest_up_value.max(up_value.abs());
        }
    }
    eprintln!(
        "[positive-control] adapter_up_tensors={up_tensor_count} largest_adapter_up_value={largest_up_value:.3e}"
    );
}

fn highest_scoring(option_logits: &[f32]) -> usize {
    let mut best_option = 0;
    for (option_index, option_logit) in option_logits.iter().enumerate() {
        if *option_logit > option_logits[best_option] {
            best_option = option_index;
        }
    }
    best_option
}

async fn prove_whole_model_learns_the_code(family: TunableHeadFamily, code_layout: CodeLayout) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let mut decision_engine = loaded_engine(family, &model_directory);
    let job = coded_job_laid_out(code_layout);
    let outcome = train_to_completion(decision_engine.as_mut(), job.clone(), "positive-control");
    print_training_report(&job, &outcome);
    print_add_on_movement(&outcome);
    let (served_accuracy, trained_accuracy) =
        held_back_accuracy(&job, &outcome, HeldBackRole::Judging);
    eprintln!(
        "[positive-control] family={family:?} judged_right_served={served_accuracy:.3} judged_right_trained={trained_accuracy:.3} parity_gap={}",
        outcome.largest_parity_gap
    );
    assert_eq!(outcome.largest_parity_gap, 0.0);
    assert!(
        trained_accuracy >= REQUIRED_TRAINED_ACCURACY,
        "the whole model must learn the code: {trained_accuracy:.3} right on unseen lines"
    );
}

#[ignore = "requires a Kev model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_learn_a_code_the_final_layers_cannot_on_kev() {
    prove_whole_model_learns_the_code(TunableHeadFamily::Kev, DEFAULT_CODE_LAYOUT).await;
}

#[ignore = "requires a Clef model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_learn_a_code_the_final_layers_cannot_on_clef() {
    prove_whole_model_learns_the_code(TunableHeadFamily::Clef, DEFAULT_CODE_LAYOUT).await;
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_learn_a_code_the_final_layers_cannot_on_laya() {
    prove_whole_model_learns_the_code(TunableHeadFamily::Laya, DEFAULT_CODE_LAYOUT).await;
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_learn_a_balanced_code_from_another_passage_on_laya() {
    prove_whole_model_learns_the_code(TunableHeadFamily::Laya, BALANCED_CODE_LAYOUT).await;
}
