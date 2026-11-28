//! One training job inside the worker, advanced a step at a time.
//!
//! - **Reading:** one example per step: the engine answers it, keeping what
//!   the final layers read when only they train.
//! - **Preparing:** the shipped final layers are read from the model
//!   folder. A final-layer training uploads what they read once; a
//!   whole-model training asks the engine for its whole model as a scorer,
//!   with add-on weights in every layer.
//! - **Checking, training, scoring:** the prepared training's own stages,
//!   each advanced a couple of seconds at a time (`training_phase`).
//!
//! No step holds the worker for long: the worker looks for a cancel only
//! between steps, so a step's length is how long an interrupted training
//! keeps running.

use std::path::Path;
use std::rc::Rc;

use decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingStage, TrainingExample, TrainingScope,
};
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{DecisionFailure, HeadInputRequest, OrderedJsonValue, QuestionOutcome};
use decision_training::head_parameters::ParameterLayout;
use decision_training::shipped_head::{ShippedHead, read_shipped_head};
use decision_training::training_settings::{
    FINAL_LAYERS_TRAINING_SETTINGS, WHOLE_MODEL_TRAINING_SETTINGS,
};
use runtime_integration::MlxRuntime;

use crate::decision_engine::DecisionTrainingStep;
use mlx_bridge::MlxArray;

#[cfg(feature = "gradient-check")]
use super::directional_gradient_check::{
    ArrayGradientCheck, GradientCheckProgress, GradientCheckStage, directional_gradient_checks,
};
use super::gpu_math::GpuMath;
use super::gpu_training::GpuTrainingSet;
use super::gpu_upload::{gpu_head_family, upload_head_inputs, upload_parameters};
use super::question_batch::OptionLayout;
use super::question_selection::{QuestionSelection, RowSource, select_questions};
use super::trainable_scorer::{FinalLayersScorer, TrainableScorer};
use super::training_phase::{PreparedTraining, TrainedLayout, TrainingPhase, gpu_failure};
use crate::decision_models::adapter_training::whole_model_layout::WholeModelLayout;

/// What a session needs from the engine it runs in.
pub trait DecisionTrainingHost {
    /// The model's answers to `questions` about `state`; with
    /// [`HeadInputRequest::Include`], also what the final layers read.
    fn read_answers(
        &mut self,
        state: &OrderedJsonValue,
        questions: &OrderedJsonValue,
        head_input_request: HeadInputRequest,
    ) -> Result<Vec<QuestionOutcome>, DecisionFailure>;
    fn mlx_runtime(&self) -> Result<&MlxRuntime, DecisionFailure>;
    fn model_directory(&self) -> &Path;
    fn tunable_family(&self) -> TunableHeadFamily;

    /// The whole model as a scorer over the questions at `row_sources`, one
    /// per training row in row order, its final layers laid out as
    /// `scorer_layout`. Families that cannot train their whole model yet
    /// refuse.
    fn whole_model_scorer(
        &mut self,
        _examples: &[TrainingExample],
        _row_sources: &[RowSource],
        _scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        Err(DecisionFailure::invalid_request(
            "this model can train only its final layers so far; train without --whole-model",
        ))
    }
}

/// A family's whole model, ready to train.
pub struct WholeModelScorer {
    pub scorer: Box<dyn TrainableScorer>,
    pub layout: Rc<WholeModelLayout>,
}

/// What preparing builds before the parity check.
struct PreparedScorer {
    scorer: Box<dyn TrainableScorer>,
    option_layout: OptionLayout,
    starting_parameters: Vec<MlxArray>,
    trained_layout: TrainedLayout,
}

/// A training job in progress.
pub struct DecisionTrainingSession {
    job: DecisionTrainingJob,
    read_examples: Vec<Vec<QuestionOutcome>>,
    training: Option<TrainingPhase>,
}

impl DecisionTrainingSession {
    pub fn new(job: DecisionTrainingJob) -> Result<Self, DecisionFailure> {
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
    pub fn advance(
        &mut self,
        host: &mut dyn DecisionTrainingHost,
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        let example_count = self.job.examples.len();
        if self.read_examples.len() < example_count {
            return self.read_next_example(host);
        }
        if self.training.is_none() {
            self.training = Some(self.prepare_training(host)?);
        }
        let Some(training) = self.training.as_mut() else {
            return Err(DecisionFailure::fatal_execution(
                "the training was not prepared",
            ));
        };
        training.advance(host.mlx_runtime()?, &self.read_examples)
    }

    fn read_next_example(
        &mut self,
        host: &mut dyn DecisionTrainingHost,
    ) -> Result<DecisionTrainingStep, DecisionFailure> {
        let example = &self.job.examples[self.read_examples.len()];
        let head_input_request = match self.job.scope {
            TrainingScope::FinalLayers => HeadInputRequest::Include,
            TrainingScope::WholeModel => HeadInputRequest::Omit,
        };
        let question_outcomes =
            host.read_answers(&example.state, &example.questions, head_input_request)?;
        if question_outcomes.len() != example.question_roles.len() {
            return Err(DecisionFailure::invalid_request(&format!(
                "example {} has {} questions but {} roles",
                self.read_examples.len() + 1,
                question_outcomes.len(),
                example.question_roles.len()
            )));
        }
        self.read_examples.push(question_outcomes);
        Ok(DecisionTrainingStep::Progress {
            stage: DecisionTrainingStage::Reading,
            completed: self.read_examples.len() as u64,
            total: self.job.examples.len() as u64,
        })
    }

    #[cfg(feature = "gradient-check")]
    /// Reads every example, prepares the training as [`Self::advance`]
    /// would, and checks the gradient of the batch of the first
    /// `checked_row_count` training rows against the loss's own slope at the
    /// starting parameters, array by array (see `directional_gradient_check`).
    /// Nothing is trained.
    ///
    /// # Errors
    ///
    /// Returns the failure of reading, preparing or any pass.
    pub fn check_starting_gradient(
        &mut self,
        host: &mut dyn DecisionTrainingHost,
        checked_row_count: usize,
        progress: &mut dyn GradientCheckProgress,
    ) -> Result<Vec<ArrayGradientCheck>, DecisionFailure> {
        let example_count = self.job.examples.len();
        while self.read_examples.len() < example_count {
            self.read_next_example(host)?;
            progress.report(
                GradientCheckStage::Reading,
                self.read_examples.len(),
                example_count,
            );
        }
        let prepared_training = self.prepared_training(host)?;
        let training_rows = &prepared_training.training_set.training_rows;
        let checked_rows = &training_rows[..checked_row_count.min(training_rows.len())];
        directional_gradient_checks(
            GpuMath {
                runtime: host.mlx_runtime()?,
            },
            &prepared_training.training_set,
            &prepared_training.starting_parameters,
            checked_rows,
            progress,
        )
        .map_err(gpu_failure)
    }

    fn prepare_training(
        &self,
        host: &mut dyn DecisionTrainingHost,
    ) -> Result<TrainingPhase, DecisionFailure> {
        let prepared_training = self.prepared_training(host)?;
        TrainingPhase::new(prepared_training, host.mlx_runtime()?)
    }

    fn prepared_training(
        &self,
        host: &mut dyn DecisionTrainingHost,
    ) -> Result<PreparedTraining, DecisionFailure> {
        let family = host.tunable_family();
        let shipped_head = match read_shipped_head(host.model_directory(), family) {
            Ok(shipped_head) => shipped_head,
            Err(read_error) => {
                return Err(DecisionFailure::fatal_execution(&read_error.to_string()));
            }
        };
        let final_layers = match self.job.scope {
            TrainingScope::FinalLayers => Some(shipped_head.trainable_head.as_ref()),
            TrainingScope::WholeModel => None,
        };
        let selection = select_questions(&self.job, &self.read_examples, final_layers)?;
        let prepared_scorer = match self.job.scope {
            TrainingScope::FinalLayers => final_layers_scorer(host, &shipped_head, &selection)?,
            TrainingScope::WholeModel => {
                whole_model_scorer(host, &self.job.examples, &shipped_head, &selection)?
            }
        };
        let settings = match self.job.scope {
            TrainingScope::FinalLayers => FINAL_LAYERS_TRAINING_SETTINGS,
            TrainingScope::WholeModel => WHOLE_MODEL_TRAINING_SETTINGS,
        };
        Ok(PreparedTraining {
            trained_layout: prepared_scorer.trained_layout,
            training_set: GpuTrainingSet {
                scorer: prepared_scorer.scorer,
                option_layout: prepared_scorer.option_layout,
                rows: selection.rows,
                training_rows: selection.training_rows,
                choosing_rows: selection.choosing_rows,
                check_rows: selection.check_rows,
            },
            starting_parameters: prepared_scorer.starting_parameters,
            served_logits: selection.served_logits,
            question_rows: selection.question_rows,
            indistinguishable_question_count: selection.indistinguishable_question_count,
            settings,
        })
    }
}

/// A final-layer training: what the final layers read, uploaded once, and
/// the layers as served.
fn final_layers_scorer(
    host: &dyn DecisionTrainingHost,
    shipped_head: &ShippedHead,
    selection: &QuestionSelection<'_>,
) -> Result<PreparedScorer, DecisionFailure> {
    let math = GpuMath {
        runtime: host.mlx_runtime()?,
    };
    let family = shipped_head.family;
    let (head_inputs, option_layout) =
        upload_head_inputs(math, family, &selection.head_inputs).map_err(gpu_failure)?;
    let layout = shipped_head.trainable_head.layout().clone();
    let starting_parameters =
        upload_parameters(math, &layout, &shipped_head.served_parameters).map_err(gpu_failure)?;
    Ok(PreparedScorer {
        scorer: Box::new(FinalLayersScorer {
            family: gpu_head_family(family, &layout),
            head_inputs,
        }),
        option_layout,
        starting_parameters,
        trained_layout: TrainedLayout::FinalLayers(layout),
    })
}

/// A whole-model training: the engine's whole model with add-ons in every
/// layer, the final layers as served and every add-on switched off.
fn whole_model_scorer(
    host: &mut dyn DecisionTrainingHost,
    examples: &[TrainingExample],
    shipped_head: &ShippedHead,
    selection: &QuestionSelection<'_>,
) -> Result<PreparedScorer, DecisionFailure> {
    let scorer_layout = shipped_head.trainable_head.layout();
    let whole_model = host.whole_model_scorer(examples, &selection.row_sources, scorer_layout)?;
    let math = GpuMath {
        runtime: host.mlx_runtime()?,
    };
    let starting_parameters = whole_model
        .layout
        .starting_parameters(math, &shipped_head.served_parameters)
        .map_err(gpu_failure)?;
    let mut option_counts = Vec::with_capacity(selection.served_logits.len());
    for served_question_logits in &selection.served_logits {
        option_counts.push(served_question_logits.len());
    }
    Ok(PreparedScorer {
        scorer: whole_model.scorer,
        option_layout: OptionLayout::new(option_counts),
        starting_parameters,
        trained_layout: TrainedLayout::WholeModel(whole_model.layout),
    })
}

/// Starts a session in an engine's slot, refusing a second one.
pub fn start_session(
    session_slot: &mut Option<DecisionTrainingSession>,
    job: DecisionTrainingJob,
) -> Result<(), DecisionFailure> {
    if session_slot.is_some() {
        return Err(DecisionFailure::invalid_request(
            "a training is already in progress",
        ));
    }
    *session_slot = Some(DecisionTrainingSession::new(job)?);
    Ok(())
}

/// Puts a session back in its slot after a step unless the step ended it.
pub fn keep_session_if_running(
    session_slot: &mut Option<DecisionTrainingSession>,
    session: DecisionTrainingSession,
    step: &Result<DecisionTrainingStep, DecisionFailure>,
) {
    if matches!(step, Ok(DecisionTrainingStep::Progress { .. })) {
        *session_slot = Some(session);
    }
}

/// Answers `questions` about `state` through `decide`.
pub fn read_through_decide(
    engine: &mut dyn crate::DecisionEngine,
    state: &OrderedJsonValue,
    questions: &OrderedJsonValue,
    head_input_request: HeadInputRequest,
) -> Result<Vec<QuestionOutcome>, DecisionFailure> {
    let parsed_questions = match decision_contract::parse_system_one_questions(questions) {
        Ok(parsed_questions) => parsed_questions,
        Err(question_error) => {
            return Err(DecisionFailure::invalid_request(
                &question_error.to_string(),
            ));
        }
    };
    let engine_output = engine.decide(state, &parsed_questions, head_input_request)?;
    Ok(engine_output.question_outcomes)
}
