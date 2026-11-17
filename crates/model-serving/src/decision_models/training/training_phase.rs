//! A prepared training, advanced a couple of seconds at a time through its
//! three stages.
//!
//! - **Checking:** a spread of questions is scored with the starting
//!   parameters, and the scores must reproduce the model's own (parity)
//!   before any weight moves. The trainable copy scores with the same code
//!   for every question, so a few dozen across the job prove what all of
//!   them would; scoring all of them through a 9B model cost over a minute.
//! - **Training:** the GPU trainer's passes, run after run.
//! - **Scoring:** the questions the caller judges with (choosing, judging
//!   and check-set questions) are scored with the best run's parameters,
//!   and the trained tensors are read back. Training questions are left
//!   out: nothing reads their trained scores, and they are most of a job.
//!
//! Every stage stops when its step's time is up and continues on the next
//! step. The worker looks for a cancel only between steps, so a stage that
//! ran to its end in one step would keep an interrupted training on the GPU
//! until it finished; found live, when a training whose caller had gone
//! held the worker through forty-second steps.
//!
//! After every step the runtime's memory is compared with where the
//! training started. A training holds a few small arrays between steps;
//! memory that keeps growing means buffers are piling up, which once pushed
//! the Mac into swapping, so the training is stopped instead.

use std::rc::Rc;
use std::time::Instant;

use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingOutcome, DecisionTrainingStage, RunEnding, RunSummary, TrainedQuestion,
    TrainedTensor,
};
use astronomical_decision_contract::{DecisionFailure, FloatVector, QuestionOutcome};
use astronomical_decision_training::head_parameters::ParameterLayout;
use astronomical_decision_training::training_settings::{SeedEnding, TrainingSettings};
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuTrainingError};
use super::gpu_training::{GpuSeedRun, GpuTrainer, GpuTrainingSet, STEP_TIME_BUDGET, TrainerStep};
use super::stepped_scoring::SteppedScoring;
use crate::decision_engine::DecisionTrainingStep;
use crate::decision_models::adapter_training::whole_model_layout::WholeModelLayout;

/// A gap this small between the model and its trainable copy is float
/// rounding from summing in another order; anything larger is a different
/// computation, and training would start from the wrong place.
const PARITY_ABSOLUTE_TOLERANCE: f32 = 2e-3;
const PARITY_RELATIVE_TOLERANCE: f32 = 1e-3;

/// How far a training's memory between steps may rise above where it
/// started. Between steps it holds the parameters, the optimizer's state
/// and each run's best parameters, well under a gigabyte even for a whole
/// model's add-ons; two gigabytes (2 × 2^30 bytes) of growth is buffers
/// piling up.
const TRAINING_MEMORY_GROWTH_LIMIT_BYTES: usize = 2 << 30;
const BYTES_PER_GIGABYTE: f64 = 1_073_741_824.0;

/// How the trained parameters map back to tensors.
pub(crate) enum TrainedLayout {
    FinalLayers(ParameterLayout),
    WholeModel(Rc<WholeModelLayout>),
}

impl TrainedLayout {
    /// The trained final layers and add-ons, from each parameter's values.
    fn exported_tensors(
        &self,
        parameter_values: Vec<Vec<f32>>,
    ) -> (Vec<TrainedTensor>, Vec<TrainedTensor>) {
        match self {
            Self::FinalLayers(layout) => {
                let flat_values: Vec<f32> = parameter_values.concat();
                (trained_tensors(layout, &flat_values), Vec::new())
            }
            Self::WholeModel(layout) => layout.exported_tensors(parameter_values),
        }
    }
}

/// Everything preparing a training produced.
pub(crate) struct PreparedTraining {
    pub(crate) trained_layout: TrainedLayout,
    pub(crate) training_set: GpuTrainingSet,
    pub(crate) starting_parameters: Vec<MlxArray>,
    /// The model's own scores for every question, in row order.
    pub(crate) served_logits: Vec<Vec<f32>>,
    /// For each example and question, its row in the training set, or
    /// `None` when training left it out.
    pub(crate) question_rows: Vec<Vec<Option<usize>>>,
    pub(crate) indistinguishable_question_count: usize,
    pub(crate) settings: TrainingSettings,
}

/// Where a prepared training stands.
enum PhaseStage {
    Checking(SteppedScoring),
    Training,
    Scoring(Box<BestRunScoring>),
}

/// The finished runs and the best one's scores so far.
struct BestRunScoring {
    runs: Vec<GpuSeedRun>,
    best_position: usize,
    scoring: SteppedScoring,
}

/// A prepared training in progress.
pub(crate) struct TrainingPhase {
    prepared: PreparedTraining,
    trainer: GpuTrainer,
    stage: PhaseStage,
    parity: ParityCheck,
    /// MLX's active and cached bytes when the training was prepared.
    starting_memory_bytes: usize,
}

impl TrainingPhase {
    pub(crate) fn new(
        prepared: PreparedTraining,
        runtime: &MlxRuntime,
    ) -> Result<Self, DecisionFailure> {
        let row_count = prepared.training_set.rows.len();
        Ok(Self {
            trainer: GpuTrainer::new(prepared.settings),
            stage: PhaseStage::Checking(SteppedScoring::new(
                parity_sample_rows(row_count),
                row_count,
            )),
            parity: ParityCheck {
                largest_gap: 0.0,
                gap_beyond_rounding: None,
            },
            starting_memory_bytes: held_memory_bytes(runtime)?,
            prepared,
        })
    }

    /// Takes one step of the current stage, a couple of seconds at most.
    pub(crate) fn advance(
        &mut self,
        runtime: &MlxRuntime,
        read_examples: &[Vec<QuestionOutcome>],
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        let math = GpuMath { runtime };
        let step_deadline = Instant::now() + STEP_TIME_BUDGET;
        let step = match std::mem::replace(&mut self.stage, PhaseStage::Training) {
            PhaseStage::Checking(parity_scoring) => {
                self.checking_step(math, parity_scoring, step_deadline)?
            }
            PhaseStage::Training => self.training_step(math)?,
            PhaseStage::Scoring(best_run_scoring) => {
                self.scoring_step(math, *best_run_scoring, step_deadline, read_examples)?
            }
        };
        check_memory_growth(held_memory_bytes(runtime)?, self.starting_memory_bytes)?;
        Ok(step)
    }

    fn checking_step(
        &mut self,
        math: GpuMath<'_>,
        mut parity_scoring: SteppedScoring,
        step_deadline: Instant,
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        parity_scoring
            .advance(
                math,
                &self.prepared.training_set,
                &self.prepared.starting_parameters,
                step_deadline,
            )
            .map_err(gpu_failure)?;
        let progress = scoring_progress(DecisionTrainingStage::Checking, &parity_scoring);
        if parity_scoring.is_finished() {
            self.parity = parity_check(&parity_scoring.into_scores(), &self.prepared.served_logits);
            self.stage = PhaseStage::Training;
        } else {
            self.stage = PhaseStage::Checking(parity_scoring);
        }
        Ok(progress)
    }

    fn training_step(
        &mut self,
        math: GpuMath<'_>,
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        let trainer_step = self
            .trainer
            .advance(
                math,
                &self.prepared.training_set,
                &self.prepared.starting_parameters,
            )
            .map_err(gpu_failure)?;
        match trainer_step {
            TrainerStep::Running {
                completed_batches,
                total_batches,
            } => Ok(DecisionTrainingStep::Progress {
                stage: DecisionTrainingStage::Training,
                completed: completed_batches,
                total: total_batches,
            }),
            TrainerStep::Finished => {
                let runs = self.trainer.take_runs();
                let Some(best_position) = best_run_position(&runs) else {
                    return Err(DecisionFailure::fatal_execution("no training run finished"));
                };
                let training_set = &self.prepared.training_set;
                let scoring = SteppedScoring::new(
                    judged_rows(
                        training_set.rows.len(),
                        &training_set.training_rows,
                        &training_set.choosing_rows,
                    ),
                    training_set.rows.len(),
                );
                let progress = scoring_progress(DecisionTrainingStage::Scoring, &scoring);
                self.stage = PhaseStage::Scoring(Box::new(BestRunScoring {
                    runs,
                    best_position,
                    scoring,
                }));
                Ok(progress)
            }
        }
    }

    fn scoring_step(
        &mut self,
        math: GpuMath<'_>,
        mut best_run_scoring: BestRunScoring,
        step_deadline: Instant,
        read_examples: &[Vec<QuestionOutcome>],
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        let best_run = &best_run_scoring.runs[best_run_scoring.best_position];
        best_run_scoring
            .scoring
            .advance(
                math,
                &self.prepared.training_set,
                &best_run.best_parameters,
                step_deadline,
            )
            .map_err(gpu_failure)?;
        if !best_run_scoring.scoring.is_finished() {
            let progress =
                scoring_progress(DecisionTrainingStage::Scoring, &best_run_scoring.scoring);
            self.stage = PhaseStage::Scoring(Box::new(best_run_scoring));
            return Ok(progress);
        }
        let mut parameter_values = Vec::with_capacity(best_run.best_parameters.len());
        for best_parameter in &best_run.best_parameters {
            parameter_values.push(math.download(best_parameter).map_err(gpu_failure)?);
        }
        let (trained_tensors, adapter_tensors) = self
            .prepared
            .trained_layout
            .exported_tensors(parameter_values);
        let outcome = DecisionTrainingOutcome {
            examples: trained_examples(
                read_examples,
                &self.prepared.question_rows,
                &best_run_scoring.scoring.into_scores(),
            ),
            trained_tensors,
            adapter_tensors,
            runs: run_summaries(&best_run_scoring.runs),
            best_seed: best_run_scoring.runs[best_run_scoring.best_position].seed,
            largest_parity_gap: self.parity.largest_gap,
            parity_gap_beyond_rounding: self.parity.gap_beyond_rounding,
            indistinguishable_question_count: self.prepared.indistinguishable_question_count,
        };
        Ok(DecisionTrainingStep::Completed(Box::new(outcome)))
    }
}

/// How many questions the parity check scores at most.
const PARITY_SAMPLE_QUESTION_COUNT: usize = 64;

/// At most [`PARITY_SAMPLE_QUESTION_COUNT`] rows spread evenly over the job,
/// the first and last included, so every part of it is represented.
fn parity_sample_rows(row_count: usize) -> Vec<usize> {
    if row_count <= PARITY_SAMPLE_QUESTION_COUNT {
        return (0..row_count).collect();
    }
    let mut sampled_rows = Vec::with_capacity(PARITY_SAMPLE_QUESTION_COUNT);
    for sample_index in 0..PARITY_SAMPLE_QUESTION_COUNT {
        sampled_rows.push(sample_index * (row_count - 1) / (PARITY_SAMPLE_QUESTION_COUNT - 1));
    }
    sampled_rows
}

/// The rows whose trained scores go back: every row that is not a
/// training row, in row order. When nothing was held back (no choosing
/// rows), the training rows go back too, since they are then what the
/// result is compared on.
fn judged_rows(row_count: usize, training_rows: &[usize], choosing_rows: &[usize]) -> Vec<usize> {
    if choosing_rows.is_empty() {
        return (0..row_count).collect();
    }
    let mut is_training_row = vec![false; row_count];
    for training_row in training_rows {
        is_training_row[*training_row] = true;
    }
    let mut rows = Vec::with_capacity(row_count - training_rows.len().min(row_count));
    for (row, is_training) in is_training_row.iter().enumerate() {
        if !is_training {
            rows.push(row);
        }
    }
    rows
}

fn scoring_progress(
    stage: DecisionTrainingStage,
    scoring: &SteppedScoring,
) -> DecisionTrainingStep {
    DecisionTrainingStep::Progress {
        stage,
        completed: scoring.scored_question_count() as u64,
        total: scoring.question_count() as u64,
    }
}

/// MLX's active and buffer-cache bytes: what the worker holds between steps.
fn held_memory_bytes(runtime: &MlxRuntime) -> Result<usize, DecisionFailure> {
    match runtime.memory_snapshot() {
        Ok(memory_snapshot) => {
            Ok(memory_snapshot.active_memory_bytes() + memory_snapshot.mlx_buffer_cache_bytes())
        }
        Err(snapshot_error) => Err(DecisionFailure::fatal_execution(&format!(
            "the training's memory could not be read: {snapshot_error}"
        ))),
    }
}

/// Refuses to go on once the memory held between steps has grown past the
/// limit; the figure is in gigabytes of 2^30 bytes, as the app shows them.
fn check_memory_growth(
    held_memory_bytes: usize,
    starting_memory_bytes: usize,
) -> Result<(), DecisionFailure> {
    let growth_bytes = held_memory_bytes.saturating_sub(starting_memory_bytes);
    if growth_bytes <= TRAINING_MEMORY_GROWTH_LIMIT_BYTES {
        return Ok(());
    }
    Err(DecisionFailure::fatal_execution(&format!(
        "the training's memory kept growing ({:.1} GB more than when it started), so it was stopped before it could slow this Mac down; nothing was trained",
        growth_bytes as f64 / BYTES_PER_GIGABYTE
    )))
}

/// How closely the trainable copy reproduced the model's scores with the
/// starting parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParityCheck {
    pub(crate) largest_gap: f32,
    /// The gap furthest past what rounding allows, when any is; training
    /// goes on and the person is told, since they decide what to do with
    /// a model that started from slightly the wrong place.
    pub(crate) gap_beyond_rounding: Option<f32>,
}

/// The largest gap between the model's scores and the trainable copy's,
/// and the worst one beyond rounding, if any.
fn parity_check(trainable_scores: &[Vec<f32>], served_logits: &[Vec<f32>]) -> ParityCheck {
    let mut largest_gap = 0.0_f32;
    // The gap furthest past what rounding allows, so the report names the
    // worst question rather than the first one over the line.
    let mut gap_beyond_rounding: Option<f32> = None;
    let mut worst_excess_ratio = 1.0_f32;
    for (trainable_question, served_question) in trainable_scores.iter().zip(served_logits) {
        for (trainable_score, served_score) in trainable_question.iter().zip(served_question) {
            let gap = (trainable_score - served_score).abs();
            let allowed =
                PARITY_ABSOLUTE_TOLERANCE + PARITY_RELATIVE_TOLERANCE * served_score.abs();
            let excess_ratio = gap / allowed;
            if excess_ratio > worst_excess_ratio || !gap.is_finite() {
                worst_excess_ratio = excess_ratio;
                gap_beyond_rounding = Some(gap);
            }
            largest_gap = largest_gap.max(gap);
        }
    }
    ParityCheck {
        largest_gap,
        gap_beyond_rounding,
    }
}

/// The position of the run with the lowest choosing loss; the first on a tie.
fn best_run_position(runs: &[GpuSeedRun]) -> Option<usize> {
    let mut best_so_far: Option<usize> = None;
    for (run_position, run) in runs.iter().enumerate() {
        let is_better = match best_so_far {
            Some(best_position) => run.best_choosing_loss < runs[best_position].best_choosing_loss,
            None => true,
        };
        if is_better {
            best_so_far = Some(run_position);
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

pub(crate) fn gpu_failure(gpu_error: GpuTrainingError) -> DecisionFailure {
    DecisionFailure::fatal_execution(&gpu_error.to_string())
}

#[cfg(test)]
#[path = "training_phase_tests.rs"]
mod tests;
