//! One head-training job inside the worker, advanced a step at a time.
//!
//! - **Reading:** one example per step: the engine answers it, keeping what
//!   the final layers read.
//! - **Preparing:** the shipped trainable layers are read from the model
//!   folder, everything is uploaded to the GPU once, and the GPU copy of the
//!   layers must reproduce the model's own scores (parity) before any
//!   weight moves.
//! - **Training:** one pass per step on the GPU, run after run.
//! - **Finishing:** the best run's scores and tensors are read back.

use std::path::Path;

use astronomical_decision_contract::head_training_job::{
    HeadTrainingJob, HeadTrainingOutcome, HeadTrainingStage, QuestionRole, RunEnding, RunSummary,
    TrainedQuestion, TrainedTensor,
};
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_decision_contract::{
    DecisionFailure, FloatVector, OrderedJsonValue, QuestionOutcome,
};
use astronomical_decision_training::head_parameters::ParameterLayout;
use astronomical_decision_training::head_training::{HEAD_TRAINING_SETTINGS, SeedEnding};
use astronomical_decision_training::shipped_head::read_shipped_head;
use astronomical_decision_training::training_losses::tempered_probabilities;
use astronomical_runtime_integration::MlxRuntime;

use crate::decision_engine::HeadTrainingStep;
use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuTrainingError};
use super::gpu_training::{
    GpuSeedRun, GpuTrainer, GpuTrainingSet, TrainerStep, TrainingRow, question_scores,
};
use super::gpu_upload::{
    download_parameters, gpu_head_family, upload_head_inputs, upload_parameters,
};

/// A gap this small between the model and the GPU copy of its final layers
/// is float rounding from summing in another order; anything larger is a
/// different computation, and training would start from the wrong place.
const PARITY_ABSOLUTE_TOLERANCE: f32 = 2e-3;
const PARITY_RELATIVE_TOLERANCE: f32 = 1e-3;

/// What a session needs from the engine it runs in.
pub(crate) trait HeadTrainingHost {
    /// The model's answers to `questions` about `state`, head inputs included.
    fn read_with_head_inputs(
        &mut self,
        state: &OrderedJsonValue,
        questions: &OrderedJsonValue,
    ) -> Result<Vec<QuestionOutcome>, DecisionFailure>;
    fn mlx_runtime(&self) -> Result<&MlxRuntime, DecisionFailure>;
    fn model_directory(&self) -> &Path;
    fn tunable_family(&self) -> TunableHeadFamily;
}

/// A training job in progress.
pub(crate) struct HeadTrainingSession {
    job: HeadTrainingJob,
    read_examples: Vec<Vec<QuestionOutcome>>,
    training: Option<TrainingPhase>,
}

/// Everything the training passes and the finish need.
struct TrainingPhase {
    layout: ParameterLayout,
    training_set: GpuTrainingSet,
    served_parameters: Vec<MlxArray>,
    trainer: GpuTrainer,
    /// For each example and question, its row in the training set, or
    /// `None` when training left it out.
    question_rows: Vec<Vec<Option<usize>>>,
    largest_parity_gap: f32,
    indistinguishable_question_count: usize,
}

impl HeadTrainingSession {
    pub(crate) fn new(job: HeadTrainingJob) -> Result<Self, DecisionFailure> {
        if job.examples.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "a training needs at least one example",
            ));
        }
        let example_count = job.examples.len();
        Ok(Self {
            job,
            read_examples: Vec::with_capacity(example_count),
            training: None,
        })
    }

    /// Takes one step.
    pub(crate) fn advance(
        &mut self,
        host: &mut dyn HeadTrainingHost,
    ) -> Result<HeadTrainingStep, DecisionFailure> {
        let example_count = self.job.examples.len();
        if self.read_examples.len() < example_count {
            return self.read_next_example(host);
        }
        if self.training.is_none() {
            self.training = Some(self.prepare_training(host)?);
        }
        let runtime = host.mlx_runtime()?;
        let math = GpuMath { runtime };
        let Some(training) = self.training.as_mut() else {
            return Err(DecisionFailure::fatal_execution(
                "the training was not prepared",
            ));
        };
        match training
            .trainer
            .advance(math, &training.training_set, &training.served_parameters)
            .map_err(gpu_failure)?
        {
            TrainerStep::Running {
                completed_epochs,
                total_epochs,
            } => Ok(HeadTrainingStep::Progress {
                stage: HeadTrainingStage::Training,
                completed: completed_epochs,
                total: total_epochs,
            }),
            TrainerStep::Finished => self.finish(math),
        }
    }

    fn read_next_example(
        &mut self,
        host: &mut dyn HeadTrainingHost,
    ) -> Result<HeadTrainingStep, DecisionFailure> {
        let example = &self.job.examples[self.read_examples.len()];
        let question_outcomes = host.read_with_head_inputs(&example.state, &example.questions)?;
        if question_outcomes.len() != example.question_roles.len() {
            return Err(DecisionFailure::invalid_request(&format!(
                "example {} has {} questions but {} roles",
                self.read_examples.len() + 1,
                question_outcomes.len(),
                example.question_roles.len()
            )));
        }
        self.read_examples.push(question_outcomes);
        Ok(HeadTrainingStep::Progress {
            stage: HeadTrainingStage::Reading,
            completed: self.read_examples.len() as u64,
            total: self.job.examples.len() as u64,
        })
    }

    fn prepare_training(
        &self,
        host: &mut dyn HeadTrainingHost,
    ) -> Result<TrainingPhase, DecisionFailure> {
        let family = host.tunable_family();
        let shipped_head = match read_shipped_head(host.model_directory(), family) {
            Ok(shipped_head) => shipped_head,
            Err(read_error) => {
                return Err(DecisionFailure::fatal_execution(&read_error.to_string()));
            }
        };
        let selection = select_questions(
            &self.job,
            &self.read_examples,
            shipped_head.trainable_head.as_ref(),
        )?;
        let math = GpuMath {
            runtime: host.mlx_runtime()?,
        };
        let (head_inputs, option_layout) =
            upload_head_inputs(math, family, &selection.head_inputs).map_err(gpu_failure)?;
        let layout = shipped_head.trainable_head.layout().clone();
        let served_parameters = upload_parameters(math, &layout, &shipped_head.served_parameters)
            .map_err(gpu_failure)?;
        let training_set = GpuTrainingSet {
            family: gpu_head_family(family, &layout),
            head_inputs,
            option_layout,
            rows: selection.rows,
            training_rows: selection.training_rows,
            choosing_rows: selection.choosing_rows,
            check_rows: selection.check_rows,
        };
        let largest_parity_gap = check_parity(
            math,
            &training_set,
            &served_parameters,
            &selection.served_logits,
        )?;
        Ok(TrainingPhase {
            layout,
            training_set,
            served_parameters,
            trainer: GpuTrainer::new(HEAD_TRAINING_SETTINGS),
            question_rows: selection.question_rows,
            largest_parity_gap,
            indistinguishable_question_count: selection.indistinguishable_question_count,
        })
    }

    fn finish(&mut self, math: GpuMath<'_>) -> Result<HeadTrainingStep, DecisionFailure> {
        let Some(training) = self.training.take() else {
            return Err(DecisionFailure::fatal_execution(
                "the training was not prepared",
            ));
        };
        let runs = training.trainer.into_runs();
        let Some(best_run) = best_run(&runs) else {
            return Err(DecisionFailure::fatal_execution("no training run finished"));
        };
        let all_rows: Vec<usize> = (0..training.training_set.rows.len()).collect();
        let trained_scores = question_scores(
            math,
            &training.training_set,
            &best_run.best_parameters,
            &all_rows,
        )
        .map_err(gpu_failure)?;
        let trained_values =
            download_parameters(math, &best_run.best_parameters).map_err(gpu_failure)?;
        let outcome = HeadTrainingOutcome {
            examples: trained_examples(
                &self.read_examples,
                &training.question_rows,
                &trained_scores,
            ),
            trained_tensors: trained_tensors(&training.layout, &trained_values),
            runs: run_summaries(&runs),
            best_seed: best_run.seed,
            largest_parity_gap: training.largest_parity_gap,
            indistinguishable_question_count: training.indistinguishable_question_count,
        };
        Ok(HeadTrainingStep::Completed(Box::new(outcome)))
    }
}

/// The questions training uses, and the host-side facts about each.
struct QuestionSelection<'read> {
    head_inputs: Vec<&'read astronomical_decision_contract::HeadInputs>,
    served_logits: Vec<Vec<f32>>,
    rows: Vec<TrainingRow>,
    training_rows: Vec<usize>,
    choosing_rows: Vec<usize>,
    check_rows: Vec<usize>,
    question_rows: Vec<Vec<Option<usize>>>,
    indistinguishable_question_count: usize,
}

fn select_questions<'read>(
    job: &HeadTrainingJob,
    read_examples: &'read [Vec<QuestionOutcome>],
    trainable_head: &dyn astronomical_decision_training::trainable_head::TrainableHead,
) -> Result<QuestionSelection<'read>, DecisionFailure> {
    let mut selection = QuestionSelection {
        head_inputs: Vec::new(),
        served_logits: Vec::new(),
        rows: Vec::new(),
        training_rows: Vec::new(),
        choosing_rows: Vec::new(),
        check_rows: Vec::new(),
        question_rows: Vec::with_capacity(read_examples.len()),
        indistinguishable_question_count: 0,
    };
    for (example, question_outcomes) in job.examples.iter().zip(read_examples) {
        let mut example_rows = Vec::with_capacity(question_outcomes.len());
        for (role, outcome) in example.question_roles.iter().zip(question_outcomes) {
            example_rows.push(select_question(
                &mut selection,
                *role,
                outcome,
                trainable_head,
            )?);
        }
        selection.question_rows.push(example_rows);
    }
    Ok(selection)
}

/// Adds one question to the selection, or leaves it out when its options
/// read alike; returns its row.
fn select_question<'read>(
    selection: &mut QuestionSelection<'read>,
    role: QuestionRole,
    outcome: &'read QuestionOutcome,
    trainable_head: &dyn astronomical_decision_training::trainable_head::TrainableHead,
) -> Result<Option<usize>, DecisionFailure> {
    let option_count = outcome.option_logits.len();
    if (outcome.input_coverage.distinguishable_option_count as usize) < option_count {
        selection.indistinguishable_question_count += usize::from(role.correct_option().is_some());
        return Ok(None);
    }
    let Some(head_inputs) = &outcome.head_inputs else {
        return Err(DecisionFailure::fatal_execution(
            "the engine returned no head inputs",
        ));
    };
    if let Err(mismatch) = trainable_head.check_inputs(head_inputs) {
        return Err(DecisionFailure::fatal_execution(&mismatch.to_string()));
    }
    if let Some(correct_option) = role.correct_option()
        && correct_option >= option_count
    {
        return Err(DecisionFailure::invalid_request(&format!(
            "question {} has {option_count} options, so option {correct_option} cannot be its right answer",
            outcome.question_name
        )));
    }
    let row = selection.rows.len();
    let temperature = outcome.applied_temperature;
    let training_row = match role {
        QuestionRole::Check => TrainingRow::Check {
            shipped_probabilities: narrowed(&tempered_probabilities(
                &outcome.option_logits,
                temperature,
            )),
            temperature,
        },
        labelled_role => TrainingRow::Labelled {
            correct_option: labelled_role.correct_option().unwrap_or(0),
            temperature,
        },
    };
    match role {
        QuestionRole::Training { .. } => selection.training_rows.push(row),
        QuestionRole::Choosing { .. } => selection.choosing_rows.push(row),
        QuestionRole::Check => selection.check_rows.push(row),
        QuestionRole::Judging { .. } => {}
    }
    selection.rows.push(training_row);
    selection.head_inputs.push(head_inputs);
    selection.served_logits.push(outcome.option_logits.clone());
    Ok(Some(row))
}

fn narrowed(probabilities: &[f64]) -> Vec<f32> {
    let mut narrowed_values = Vec::with_capacity(probabilities.len());
    for probability in probabilities {
        narrowed_values.push(*probability as f32);
    }
    narrowed_values
}

/// The largest gap between the model's scores and the GPU copy's with the
/// shipped weights; refuses a gap beyond rounding.
fn check_parity(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    served_parameters: &[MlxArray],
    served_logits: &[Vec<f32>],
) -> Result<f32, DecisionFailure> {
    let all_rows: Vec<usize> = (0..served_logits.len()).collect();
    let gpu_scores =
        question_scores(math, training_set, served_parameters, &all_rows).map_err(gpu_failure)?;
    let mut largest_gap = 0.0_f32;
    for (gpu_question, served_question) in gpu_scores.iter().zip(served_logits) {
        for (gpu_score, served_score) in gpu_question.iter().zip(served_question) {
            let gap = (gpu_score - served_score).abs();
            let allowed =
                PARITY_ABSOLUTE_TOLERANCE + PARITY_RELATIVE_TOLERANCE * served_score.abs();
            let is_close = gap <= allowed;
            if !is_close {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "the GPU copy of the final layers scores {gap:.5} away from the model, so training would start from the wrong place; nothing was trained"
                )));
            }
            largest_gap = largest_gap.max(gap);
        }
    }
    Ok(largest_gap)
}

/// The run with the lowest choosing loss; the first on a tie.
fn best_run(runs: &[GpuSeedRun]) -> Option<&GpuSeedRun> {
    let mut best_so_far: Option<&GpuSeedRun> = None;
    for run in runs {
        let is_better = match best_so_far {
            Some(current_best) => run.best_choosing_loss < current_best.best_choosing_loss,
            None => true,
        };
        if is_better {
            best_so_far = Some(run);
        }
    }
    best_so_far
}

fn trained_examples(
    read_examples: &[Vec<QuestionOutcome>],
    question_rows: &[Vec<Option<usize>>],
    trained_scores: &[Vec<f32>],
) -> Vec<Vec<TrainedQuestion>> {
    let mut examples = Vec::with_capacity(read_examples.len());
    for (question_outcomes, example_rows) in read_examples.iter().zip(question_rows) {
        let mut trained_questions = Vec::with_capacity(question_outcomes.len());
        for (outcome, row) in question_outcomes.iter().zip(example_rows) {
            let mut served = outcome.clone();
            // The inputs stay in the worker; only answers and scores go back.
            served.head_inputs = None;
            let trained_logits = match row {
                Some(row) => trained_scores[*row].clone(),
                None => Vec::new(),
            };
            trained_questions.push(TrainedQuestion {
                served,
                trained_logits,
            });
        }
        examples.push(trained_questions);
    }
    examples
}

fn trained_tensors(layout: &ParameterLayout, trained_values: &[f32]) -> Vec<TrainedTensor> {
    let mut tensors = Vec::with_capacity(layout.tensors.len());
    for tuned_tensor in layout.head_tuning(trained_values).tensors {
        tensors.push(TrainedTensor {
            name: tuned_tensor.name,
            shape: tuned_tensor.shape,
            values: FloatVector(tuned_tensor.values),
        });
    }
    tensors
}

fn run_summaries(runs: &[GpuSeedRun]) -> Vec<RunSummary> {
    let mut summaries = Vec::with_capacity(runs.len());
    for run in runs {
        summaries.push(RunSummary {
            seed: run.seed,
            best_choosing_loss: run.best_choosing_loss,
            best_epoch: run.best_epoch,
            epochs_run: run.epochs_run,
            ending: match run.ending {
                SeedEnding::StoppedImproving => RunEnding::StoppedImproving,
                SeedEnding::ReachedEpochLimit => RunEnding::ReachedEpochLimit,
                SeedEnding::NonFiniteStep { epoch } => RunEnding::NonFiniteStep { epoch },
            },
        });
    }
    summaries
}

fn gpu_failure(gpu_error: GpuTrainingError) -> DecisionFailure {
    DecisionFailure::fatal_execution(&gpu_error.to_string())
}

/// Starts a session in an engine's slot, refusing a second one.
pub(crate) fn start_session(
    session_slot: &mut Option<HeadTrainingSession>,
    job: HeadTrainingJob,
) -> Result<(), DecisionFailure> {
    if session_slot.is_some() {
        return Err(DecisionFailure::invalid_request(
            "a head training is already in progress",
        ));
    }
    *session_slot = Some(HeadTrainingSession::new(job)?);
    Ok(())
}

/// Puts a session back in its slot after a step unless the step ended it.
pub(crate) fn keep_session_if_running(
    session_slot: &mut Option<HeadTrainingSession>,
    session: HeadTrainingSession,
    step: &Result<HeadTrainingStep, DecisionFailure>,
) {
    if matches!(step, Ok(HeadTrainingStep::Progress { .. })) {
        *session_slot = Some(session);
    }
}

/// Answers `questions` about `state` with head inputs, through `decide`.
pub(crate) fn read_through_decide(
    engine: &mut dyn crate::DecisionEngine,
    state: &OrderedJsonValue,
    questions: &OrderedJsonValue,
) -> Result<Vec<QuestionOutcome>, DecisionFailure> {
    let parsed_questions =
        match astronomical_decision_contract::parse_system_one_questions(questions) {
            Ok(parsed_questions) => parsed_questions,
            Err(question_error) => {
                return Err(DecisionFailure::invalid_request(
                    &question_error.to_string(),
                ));
            }
        };
    let engine_output = engine.decide(
        state,
        &parsed_questions,
        astronomical_decision_contract::HeadInputRequest::Include,
    )?;
    Ok(engine_output.question_outcomes)
}

#[cfg(test)]
#[path = "training_session_tests.rs"]
mod tests;
