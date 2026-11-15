//! Training a family's final layers on the GPU, several runs over.
//!
//! The loop is `astronomical-decision-training`'s `head_training`, run on
//! MLX arrays: each run starts from the served weights, walks the training
//! questions in shuffled batches mixed with check-set questions, keeps the
//! weights from its best pass on the choosing questions, and stops once
//! several passes bring no improvement. The host only shuffles question
//! numbers, builds each batch's small target matrix, and reads back one loss
//! per pass and one gradient length per step.

#[cfg(test)]
use astronomical_decision_training::head_training::TrainingProgress;
use astronomical_decision_training::head_training::{SeedEnding, TrainingSettings, shuffled_order};
use astronomical_decision_training::training_losses::LABEL_SMOOTHING;
use mlx_bridge::MlxArray;

use super::gpu_adam::{GpuAdam, StepOutcome};
use super::gpu_heads::{GpuHeadFamily, GpuHeadInputs};
use super::gpu_math::{GpuMath, GpuResult};
use super::question_batch::{OptionLayout, QuestionBatch};
use super::row_objective::{HostRow, RowTargets, mean_cross_entropy, option_score_gradients};

/// What the host knows about one question row of a training.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TrainingRow {
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
pub(crate) struct GpuTrainingSet {
    pub(crate) family: GpuHeadFamily,
    pub(crate) head_inputs: GpuHeadInputs,
    pub(crate) option_layout: OptionLayout,
    pub(crate) rows: Vec<TrainingRow>,
    pub(crate) training_rows: Vec<usize>,
    pub(crate) choosing_rows: Vec<usize>,
    pub(crate) check_rows: Vec<usize>,
}

/// One run's best weights, still on the GPU, and how it got there.
pub(crate) struct GpuSeedRun {
    pub(crate) seed: u64,
    pub(crate) best_parameters: Vec<MlxArray>,
    pub(crate) best_choosing_loss: f64,
    pub(crate) best_epoch: usize,
    pub(crate) epochs_run: usize,
    pub(crate) ending: SeedEnding,
}

/// Where a stepped training stands after one step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TrainerStep {
    /// `completed` of at most `total` passes are done.
    Running {
        completed_epochs: u64,
        total_epochs: u64,
    },
    /// Every run has ended.
    Finished,
}

/// The run in progress.
struct SeedState {
    parameters: Vec<MlxArray>,
    optimizer: GpuAdam,
    run: GpuSeedRun,
}

/// A training advanced one pass at a time, so the worker can report
/// progress and stay responsive between passes.
pub(crate) struct GpuTrainer {
    settings: TrainingSettings,
    next_seed_position: usize,
    current_seed: Option<SeedState>,
    finished_runs: Vec<GpuSeedRun>,
}

impl GpuTrainer {
    pub(crate) fn new(settings: TrainingSettings) -> Self {
        Self {
            settings,
            next_seed_position: 0,
            current_seed: None,
            finished_runs: Vec::with_capacity(settings.seeds.len()),
        }
    }

    fn total_epochs(&self) -> u64 {
        (self.settings.seeds.len() * self.settings.maximum_epochs) as u64
    }

    /// Starts the next run, or trains the current run one pass.
    pub(crate) fn advance(
        &mut self,
        math: GpuMath<'_>,
        training_set: &GpuTrainingSet,
        served_parameters: &[MlxArray],
    ) -> GpuResult<TrainerStep> {
        let Some(mut seed_state) = self.current_seed.take() else {
            return self.start_next_seed(math, training_set, served_parameters);
        };
        let settings = self.settings;
        let epoch = seed_state.run.epochs_run + 1;
        seed_state.run.epochs_run = epoch;
        let seed = seed_state.run.seed;
        let pass_outcome = train_one_pass(
            math,
            (training_set, served_parameters, settings),
            (&mut seed_state.parameters, &mut seed_state.optimizer),
            (seed, epoch),
        )?;
        if pass_outcome == StepOutcome::RefusedNonFinite {
            seed_state.run.ending = SeedEnding::NonFiniteStep { epoch };
            return Ok(self.finish_seed(seed_state));
        }
        let epoch_loss = choosing_loss(math, training_set, &seed_state.parameters)?;
        if epoch_loss < seed_state.run.best_choosing_loss {
            seed_state.run.best_choosing_loss = epoch_loss;
            seed_state.run.best_parameters = copied(math, &seed_state.parameters)?;
            seed_state.run.best_epoch = epoch;
        }
        if epoch - seed_state.run.best_epoch >= settings.patience_epochs {
            seed_state.run.ending = SeedEnding::StoppedImproving;
            return Ok(self.finish_seed(seed_state));
        }
        if epoch >= settings.maximum_epochs {
            return Ok(self.finish_seed(seed_state));
        }
        let epoch_offset = ((self.next_seed_position - 1) * settings.maximum_epochs) as u64;
        self.current_seed = Some(seed_state);
        Ok(TrainerStep::Running {
            completed_epochs: epoch_offset + epoch as u64,
            total_epochs: self.total_epochs(),
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
        let run = GpuSeedRun {
            seed,
            best_choosing_loss: choosing_loss(math, training_set, &parameters)?,
            best_parameters: copied(math, &parameters)?,
            best_epoch: 0,
            epochs_run: 0,
            ending: SeedEnding::ReachedEpochLimit,
        };
        let epoch_offset = (self.next_seed_position * self.settings.maximum_epochs) as u64;
        self.next_seed_position += 1;
        self.current_seed = Some(SeedState {
            parameters,
            optimizer,
            run,
        });
        Ok(TrainerStep::Running {
            completed_epochs: epoch_offset,
            total_epochs: self.total_epochs(),
        })
    }

    /// Ends the current run; the count jumps to the end of its passes, so
    /// the time left shrinks rather than grows when a run stops early.
    fn finish_seed(&mut self, seed_state: SeedState) -> TrainerStep {
        self.finished_runs.push(seed_state.run);
        TrainerStep::Running {
            completed_epochs: (self.next_seed_position * self.settings.maximum_epochs) as u64,
            total_epochs: self.total_epochs(),
        }
    }

    pub(crate) fn into_runs(self) -> Vec<GpuSeedRun> {
        self.finished_runs
    }
}

/// Runs every seed of `settings` to the end, reporting each pass; the
/// worker steps a `GpuTrainer` itself, so only tests run it whole.
#[cfg(test)]
pub(crate) fn train_seeds_on_gpu(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    served_parameters: &[MlxArray],
    settings: TrainingSettings,
    training_progress: &mut dyn TrainingProgress,
) -> GpuResult<Vec<GpuSeedRun>> {
    let mut trainer = GpuTrainer::new(settings);
    while let TrainerStep::Running {
        completed_epochs,
        total_epochs,
    } = trainer.advance(math, training_set, served_parameters)?
    {
        training_progress.report(completed_epochs, total_epochs);
    }
    Ok(trainer.into_runs())
}

/// One shuffled pass over the training questions; stops at the first refused step.
fn train_one_pass(
    math: GpuMath<'_>,
    (training_set, served_parameters, settings): (&GpuTrainingSet, &[MlxArray], TrainingSettings),
    (parameters, optimizer): (&mut [MlxArray], &mut GpuAdam),
    (seed, epoch): (u64, usize),
) -> GpuResult<StepOutcome> {
    let question_order = shuffled_order(training_set.training_rows.len(), seed, epoch as u64);
    let check_order = shuffled_order(training_set.check_rows.len(), seed ^ 0x5eed, epoch as u64);
    let check_count = settings.check_questions_per_batch.min(check_order.len());
    let mut check_cursor = 0;
    for batch_positions in question_order.chunks(settings.batch_question_count.max(1)) {
        let mut batch_question_rows = Vec::with_capacity(batch_positions.len() + check_count);
        let mut row_weights = Vec::with_capacity(batch_question_rows.capacity());
        for question_position in batch_positions {
            batch_question_rows.push(training_set.training_rows[*question_position]);
            row_weights.push(1.0 / batch_positions.len() as f64);
        }
        for _ in 0..check_count {
            batch_question_rows
                .push(training_set.check_rows[check_order[check_cursor % check_order.len()]]);
            row_weights.push(settings.check_weight / check_count as f64);
            check_cursor += 1;
        }
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
            return Ok(StepOutcome::RefusedNonFinite);
        }
    }
    Ok(StepOutcome::Taken)
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
    let family = training_set.family;
    let option_scores =
        family.option_scores(math, parameters, &training_set.head_inputs, &question_batch)?;
    let score_gradients =
        option_score_gradients(math, &option_scores, &question_batch, &row_targets)?;
    family.parameter_gradients(
        math,
        parameters,
        &training_set.head_inputs,
        &question_batch,
        &score_gradients,
    )
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
fn choosing_loss(
    math: GpuMath<'_>,
    training_set: &GpuTrainingSet,
    parameters: &[MlxArray],
) -> GpuResult<f64> {
    if training_set.choosing_rows.is_empty() {
        return Ok(f64::INFINITY);
    }
    let batch_rows = training_set
        .option_layout
        .batch_rows(&training_set.choosing_rows);
    let question_batch = QuestionBatch::upload(math, &batch_rows)?;
    let mut targets = Vec::with_capacity(training_set.choosing_rows.len());
    for question_row in &training_set.choosing_rows {
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
    let option_scores = training_set.family.option_scores(
        math,
        parameters,
        &training_set.head_inputs,
        &question_batch,
    )?;
    let mean_loss = mean_cross_entropy(math, &option_scores, &question_batch, &row_targets)?;
    Ok(if mean_loss.is_finite() {
        mean_loss
    } else {
        f64::INFINITY
    })
}

/// Each listed question's option scores under `parameters`, read back.
pub(crate) fn question_scores(
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
    let option_scores = training_set.family.option_scores(
        math,
        parameters,
        &training_set.head_inputs,
        &question_batch,
    )?;
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
