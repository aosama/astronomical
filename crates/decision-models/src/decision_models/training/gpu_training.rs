//! Training a family's final layers on the GPU, several runs over.
//!
//! The loop is `decision-training`'s `training_settings`, run on
//! MLX arrays: each run starts from the served weights, walks the training
//! questions in shuffled batches mixed with check-set questions, keeps the
//! weights from its best pass on the choosing questions, and stops once
//! several passes bring no improvement. The host only shuffles question
//! numbers, builds each batch's small target matrix, and reads back one loss
//! per pass and one gradient length per step.

use decision_training::training_losses::LABEL_SMOOTHING;
#[cfg(test)]
use decision_training::training_settings::TrainingProgress;
use decision_training::training_settings::{SeedEnding, TrainingSettings, shuffled_order};
use std::time::{Duration, Instant};

use mlx_bridge::MlxArray;

use super::gpu_adam::{GpuAdam, StepOutcome};
use super::gpu_math::{GpuMath, GpuResult};
use super::question_batch::{OptionLayout, QuestionBatch};
use super::row_objective::{HostRow, RowTargets, mean_cross_entropy};
use super::trainable_scorer::TrainableScorer;

/// What the host knows about one question row of a training.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingRow {
    /// A labelled question: its right option and serving temperature.
    Labelled {
        correct_option: usize,
        temperature: f64,
    },
    /// A check-set question: the shipped probabilities training must keep.
    Check {
        shipped_probabilities: Vec<f32>,
        temperature: f64,
    },
}

/// Every question of one training, uploaded once, and which part each is in.
pub struct GpuTrainingSet {
    pub scorer: Box<dyn TrainableScorer>,
    pub option_layout: OptionLayout,
    pub rows: Vec<TrainingRow>,
    pub training_rows: Vec<usize>,
    pub choosing_rows: Vec<usize>,
    pub check_rows: Vec<usize>,
}

/// One run's best weights, still on the GPU, and how it got there.
pub struct GpuSeedRun {
    pub seed: u64,
    pub best_parameters: Vec<MlxArray>,
    pub best_choosing_loss: f64,
    pub best_epoch: usize,
    pub epochs_run: usize,
    pub ending: SeedEnding,
}

/// Where a stepped training stands after one step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainerStep {
    /// `completed` of at most `total` optimizer steps (batches) are done.
    /// Steps, not passes: a whole-model pass takes most of a minute, and a
    /// count that stands still that long leaves the caller's time-left
    /// estimate drifting upward until the pass ends.
    Running {
        completed_batches: u64,
        total_batches: u64,
    },
    /// Every run has ended.
    Finished,
}

/// The run in progress.
struct SeedState {
    parameters: Vec<MlxArray>,
    optimizer: GpuAdam,
    run: GpuSeedRun,
    /// Where the current pass stands, or `None` between passes.
    pass: Option<PassCursor>,
}

/// How far one pass over the training questions has got.
struct PassCursor {
    question_order: Vec<usize>,
    check_order: Vec<usize>,
    next_batch: usize,
    check_cursor: usize,
}

/// Where a pass stood when a step ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PassProgress {
    Completed,
    /// The step's time ran out; the pass continues next step.
    Interrupted,
    /// A batch's gradient held a non-finite number.
    Refused,
}

/// How long one step may keep working before it returns to the worker.
/// The worker looks for a cancel only between steps, so this is how long
/// an interrupted training keeps the GPU, plus the batch in flight (about a
/// second and a half for Laya's whole model, measured). The daemon also gives up on a step that sends
/// no progress for a minute, and a whole-model pass over a thousand
/// questions takes longer than that; a final-layer pass takes milliseconds
/// and always finishes in one step.
pub const STEP_TIME_BUDGET: Duration = Duration::from_secs(2);

/// A training advanced a few seconds at a time, so the worker can report
/// progress and stay responsive between steps.
pub struct GpuTrainer {
    settings: TrainingSettings,
    step_time_budget: Duration,
    next_seed_position: usize,
    current_seed: Option<SeedState>,
    finished_runs: Vec<GpuSeedRun>,
    /// The choosing loss of the starting parameters, the same for every
    /// run, so it is measured once.
    starting_choosing_loss: Option<f64>,
}

impl GpuTrainer {
    pub fn new(settings: TrainingSettings) -> Self {
        Self::with_step_time_budget(settings, STEP_TIME_BUDGET)
    }

    /// A trainer whose steps stop after `step_time_budget`; tests pass zero
    /// to stop after every batch.
    pub fn with_step_time_budget(settings: TrainingSettings, step_time_budget: Duration) -> Self {
        Self {
            settings,
            step_time_budget,
            next_seed_position: 0,
            current_seed: None,
            finished_runs: Vec::with_capacity(settings.seeds.len()),
            starting_choosing_loss: None,
        }
    }

    /// Batches in one pass over the training questions; at least one, so
    /// the count still moves when there is nothing to train on.
    fn batches_per_pass(&self, training_set: &GpuTrainingSet) -> u64 {
        let batch_size = self.settings.batch_question_count.max(1);
        (training_set.training_rows.len().div_ceil(batch_size)).max(1) as u64
    }

    /// The batches done once `finished_run_count` runs have ended and the
    /// current run has finished `finished_pass_count` passes.
    fn batches_before(
        &self,
        training_set: &GpuTrainingSet,
        finished_run_count: usize,
        finished_pass_count: usize,
    ) -> u64 {
        (finished_run_count * self.settings.maximum_epochs + finished_pass_count) as u64
            * self.batches_per_pass(training_set)
    }

    fn total_batches(&self, training_set: &GpuTrainingSet) -> u64 {
        self.batches_before(training_set, self.settings.seeds.len(), 0)
    }

    /// Starts the next run, or trains the current run for up to
    /// [`STEP_TIME_BUDGET`].
    pub fn advance(
        &mut self,
        math: GpuMath<'_>,
        training_set: &GpuTrainingSet,
        served_parameters: &[MlxArray],
    ) -> GpuResult<TrainerStep> {
        let Some(mut seed_state) = self.current_seed.take() else {
            return self.start_next_seed(math, training_set, served_parameters);
        };
        if seed_state.pass.is_none() {
            seed_state.run.epochs_run += 1;
            seed_state.pass = Some(PassCursor::new(
                training_set,
                seed_state.run.seed,
                seed_state.run.epochs_run,
            ));
        }
        let epoch = seed_state.run.epochs_run;
        let pass_progress = continue_pass(
            math,
            (training_set, served_parameters, self.settings),
            &mut seed_state,
            Instant::now() + self.step_time_budget,
        )?;
        match pass_progress {
            PassProgress::Interrupted => {
                let batches_into_pass = match &seed_state.pass {
                    Some(pass_cursor) => pass_cursor.next_batch as u64,
                    None => 0,
                };
                let completed_batches =
                    self.batches_before(training_set, self.next_seed_position - 1, epoch - 1)
                        + batches_into_pass;
                self.current_seed = Some(seed_state);
                Ok(TrainerStep::Running {
                    completed_batches,
                    total_batches: self.total_batches(training_set),
                })
            }
            PassProgress::Refused => {
                seed_state.run.ending = SeedEnding::NonFiniteStep { epoch };
                Ok(self.finish_seed(training_set, seed_state))
            }
            PassProgress::Completed => {
                seed_state.pass = None;
                self.end_pass(math, training_set, seed_state)
            }
        }
    }

    /// Scores the finished pass on the choosing questions, keeps the best
    /// weights, and decides whether the run goes on.
    fn end_pass(
        &mut self,
        math: GpuMath<'_>,
        training_set: &GpuTrainingSet,
        mut seed_state: SeedState,
    ) -> GpuResult<TrainerStep> {
        let settings = self.settings;
        let epoch = seed_state.run.epochs_run;
        let epoch_loss = choosing_loss(math, training_set, &seed_state.parameters)?;
        if epoch_loss < seed_state.run.best_choosing_loss {
            seed_state.run.best_choosing_loss = epoch_loss;
            seed_state.run.best_parameters = copied(math, &seed_state.parameters)?;
            seed_state.run.best_epoch = epoch;
        }
        if epoch - seed_state.run.best_epoch >= settings.patience_epochs {
            seed_state.run.ending = SeedEnding::StoppedImproving;
            return Ok(self.finish_seed(training_set, seed_state));
        }
        if epoch >= settings.maximum_epochs {
            return Ok(self.finish_seed(training_set, seed_state));
        }
        self.current_seed = Some(seed_state);
        Ok(TrainerStep::Running {
            completed_batches: self.batches_before(
                training_set,
                self.next_seed_position - 1,
                epoch,
            ),
            total_batches: self.total_batches(training_set),
        })
    }

    fn start_next_seed(
        &mut self,
        math: GpuMath<'_>,
        training_set: &GpuTrainingSet,
        served_parameters: &[MlxArray],
    ) -> GpuResult<TrainerStep> {
        let Some(seed) = self.settings.seeds.get(self.next_seed_position).copied() else {
            return Ok(TrainerStep::Finished);
        };
        let parameters = copied(math, served_parameters)?;
        let optimizer = GpuAdam::new(math, self.settings.adam, &parameters)?;
        let starting_choosing_loss = match self.starting_choosing_loss {
            Some(starting_choosing_loss) => starting_choosing_loss,
            None => {
                let measured_loss = choosing_loss(math, training_set, &parameters)?;
                self.starting_choosing_loss = Some(measured_loss);
                measured_loss
            }
        };
        let run = GpuSeedRun {
            seed,
            best_choosing_loss: starting_choosing_loss,
            best_parameters: copied(math, &parameters)?,
            best_epoch: 0,
            epochs_run: 0,
            ending: SeedEnding::ReachedEpochLimit,
        };
        let completed_batches = self.batches_before(training_set, self.next_seed_position, 0);
        self.next_seed_position += 1;
        self.current_seed = Some(SeedState {
            parameters,
            optimizer,
            run,
            pass: None,
        });
        Ok(TrainerStep::Running {
            completed_batches,
            total_batches: self.total_batches(training_set),
        })
    }

    /// Ends the current run; the count jumps to the end of its steps, so
    /// the time left shrinks rather than grows when a run stops early.
    fn finish_seed(&mut self, training_set: &GpuTrainingSet, seed_state: SeedState) -> TrainerStep {
        self.finished_runs.push(seed_state.run);
        TrainerStep::Running {
            completed_batches: self.batches_before(training_set, self.next_seed_position, 0),
            total_batches: self.total_batches(training_set),
        }
    }

    /// The finished runs, leaving the trainer with none.
    pub fn take_runs(&mut self) -> Vec<GpuSeedRun> {
        std::mem::take(&mut self.finished_runs)
    }
}

/// Runs every seed of `settings` to the end, reporting each pass; the
/// worker steps a `GpuTrainer` itself, so only tests run it whole.
#[cfg(test)]
pub fn train_seeds_on_gpu(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    served_parameters: &[MlxArray],
    settings: TrainingSettings,
    training_progress: &mut dyn TrainingProgress,
) -> GpuResult<Vec<GpuSeedRun>> {
    let mut trainer = GpuTrainer::new(settings);
    while let TrainerStep::Running {
        completed_batches,
        total_batches,
    } = trainer.advance(math, training_set, served_parameters)?
    {
        training_progress.report(completed_batches, total_batches);
    }
    Ok(trainer.take_runs())
}

impl PassCursor {
    fn new(training_set: &GpuTrainingSet, seed: u64, epoch: usize) -> Self {
        Self {
            question_order: shuffled_order(training_set.training_rows.len(), seed, epoch as u64),
            check_order: shuffled_order(training_set.check_rows.len(), seed ^ 0x5eed, epoch as u64),
            next_batch: 0,
            check_cursor: 0,
        }
    }

    /// The next batch's question rows and weights: its training questions,
    /// sharing a weight of one, then the check-set questions, sharing
    /// `check_weight`. `None` once the pass has no batch left.
    fn next_batch(
        &mut self,
        training_set: &GpuTrainingSet,
        settings: TrainingSettings,
    ) -> Option<(Vec<usize>, Vec<f64>)> {
        let batch_size = settings.batch_question_count.max(1);
        let batch_start = self.next_batch * batch_size;
        if batch_start >= self.question_order.len() {
            return None;
        }
        let batch_end = (batch_start + batch_size).min(self.question_order.len());
        let batch_positions = &self.question_order[batch_start..batch_end];
        let check_count = settings
            .check_questions_per_batch
            .min(self.check_order.len());
        let mut batch_question_rows = Vec::with_capacity(batch_positions.len() + check_count);
        let mut row_weights = Vec::with_capacity(batch_question_rows.capacity());
        for question_position in batch_positions {
            batch_question_rows.push(training_set.training_rows[*question_position]);
            row_weights.push(1.0 / batch_positions.len() as f64);
        }
        for _ in 0..check_count {
            let check_position = self.check_order[self.check_cursor % self.check_order.len()];
            batch_question_rows.push(training_set.check_rows[check_position]);
            row_weights.push(settings.check_weight / check_count as f64);
            self.check_cursor += 1;
        }
        self.next_batch += 1;
        Some((batch_question_rows, row_weights))
    }
}

/// Trains the run's current pass batch by batch until the pass ends, a
/// step is refused, or `step_deadline` passes.
fn continue_pass(
    math: GpuMath<'_>,
    (training_set, served_parameters, settings): (&GpuTrainingSet, &[MlxArray], TrainingSettings),
    seed_state: &mut SeedState,
    step_deadline: Instant,
) -> GpuResult<PassProgress> {
    let SeedState {
        parameters,
        optimizer,
        pass,
        ..
    } = seed_state;
    let Some(pass_cursor) = pass.as_mut() else {
        return Ok(PassProgress::Completed);
    };
    while let Some((batch_question_rows, row_weights)) =
        pass_cursor.next_batch(training_set, settings)
    {
        let gradients = batch_gradients(
            math,
            training_set,
            parameters,
            &batch_question_rows,
            &row_weights,
        )?;
        if optimizer.step(math, parameters, &gradients, served_parameters)?
            == StepOutcome::RefusedNonFinite
        {
            return Ok(PassProgress::Refused);
        }
        let pass_has_more = pass_cursor.next_batch * settings.batch_question_count.max(1)
            < pass_cursor.question_order.len();
        if pass_has_more && Instant::now() >= step_deadline {
            return Ok(PassProgress::Interrupted);
        }
    }
    Ok(PassProgress::Completed)
}

/// The batch loss's gradient with respect to every parameter.
fn batch_gradients(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    parameters: &[MlxArray],
    batch_question_rows: &[usize],
    row_weights: &[f64],
) -> GpuResult<Vec<MlxArray>> {
    let batch_rows = training_set.option_layout.batch_rows(batch_question_rows);
    let question_batch = QuestionBatch::upload(math, &batch_rows)?;
    let mut targets = Vec::with_capacity(batch_question_rows.len());
    for question_row in batch_question_rows {
        targets.push(row_target(training_set, *question_row, LABEL_SMOOTHING));
    }
    let mut host_rows = Vec::with_capacity(targets.len());
    for ((target, temperature), weight) in targets.iter().zip(row_weights) {
        host_rows.push(HostRow {
            target,
            temperature: *temperature,
            weight: *weight,
        });
    }
    let row_targets = RowTargets::upload(math, &host_rows, question_batch.widest_option_count)?;
    training_set
        .scorer
        .parameter_gradients(math, parameters, &question_batch, &row_targets)
}

/// A row's target distribution and temperature: a labelled row's answer,
/// smoothed by `smoothing`, or a check row's shipped probabilities.
fn row_target(
    training_set: &GpuTrainingSet,
    question_row: usize,
    smoothing: f64,
) -> (Vec<f32>, f64) {
    let option_count = training_set.option_layout.option_count(question_row);
    match &training_set.rows[question_row] {
        TrainingRow::Labelled {
            correct_option,
            temperature,
        } => {
            let mut target = vec![(smoothing / option_count as f64) as f32; option_count];
            target[*correct_option] += (1.0 - smoothing) as f32;
            (target, *temperature)
        }
        TrainingRow::Check {
            shipped_probabilities,
            temperature,
        } => (shipped_probabilities.clone(), *temperature),
    }
}

/// Mean surprise at the right answer over the choosing questions, at the
/// temperature each is served with.
///
/// With too few different texts to hold any back there are no choosing
/// questions, and the training questions stand in: the run then stops when
/// it stops improving on what it learns from, or at its epoch limit. An
/// empty set would leave the loss infinite at every epoch, so the starting
/// weights would always look best and come back untrained.
fn choosing_loss(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    parameters: &[MlxArray],
) -> GpuResult<f64> {
    let choosing_rows = if training_set.choosing_rows.is_empty() {
        &training_set.training_rows
    } else {
        &training_set.choosing_rows
    };
    if choosing_rows.is_empty() {
        return Ok(f64::INFINITY);
    }
    let batch_rows = training_set.option_layout.batch_rows(choosing_rows);
    let question_batch = QuestionBatch::upload(math, &batch_rows)?;
    let mut targets = Vec::with_capacity(choosing_rows.len());
    for question_row in choosing_rows {
        targets.push(row_target(training_set, *question_row, 0.0));
    }
    let mut host_rows = Vec::with_capacity(targets.len());
    for (target, temperature) in &targets {
        host_rows.push(HostRow {
            target,
            temperature: *temperature,
            weight: 1.0,
        });
    }
    let row_targets = RowTargets::upload(math, &host_rows, question_batch.widest_option_count)?;
    let option_scores = training_set
        .scorer
        .option_scores(math, parameters, &question_batch)?;
    let mean_loss = mean_cross_entropy(math, &option_scores, &question_batch, &row_targets)?;
    Ok(if mean_loss.is_finite() {
        mean_loss
    } else {
        f64::INFINITY
    })
}

/// Each listed question's option scores under `parameters`, read back.
pub fn question_scores(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    parameters: &[MlxArray],
    question_rows: &[usize],
) -> GpuResult<Vec<Vec<f32>>> {
    if question_rows.is_empty() {
        return Ok(Vec::new());
    }
    let batch_rows = training_set.option_layout.batch_rows(question_rows);
    let question_batch = QuestionBatch::upload(math, &batch_rows)?;
    let option_scores = training_set
        .scorer
        .option_scores(math, parameters, &question_batch)?;
    let flat_scores = math.download(&option_scores)?;
    let mut per_question = Vec::with_capacity(question_rows.len());
    let mut next_option = 0;
    for question_row in question_rows {
        let option_count = training_set.option_layout.option_count(*question_row);
        per_question.push(flat_scores[next_option..next_option + option_count].to_vec());
        next_option += option_count;
    }
    Ok(per_question)
}

/// Independent copies, so later steps cannot change a kept snapshot.
fn copied(math: GpuMath<'_>, parameters: &[MlxArray]) -> GpuResult<Vec<MlxArray>> {
    let mut copies = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        copies.push(math.runtime.copy(parameter)?);
    }
    let mut copy_references = Vec::with_capacity(copies.len());
    for copy in &copies {
        copy_references.push(copy);
    }
    math.runtime.evaluate_arrays(&copy_references)?;
    Ok(copies)
}

#[cfg(test)]
#[path = "gpu_training_tests.rs"]
mod tests;
