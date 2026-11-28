//! The [`DecisionEngine`] every decision family runs behind: the folder it is
//! bound to, the MLX memory limits it is granted, the cleanup and memory
//! report after each request, and any training in progress. A family supplies
//! only what differs, through [`DecisionFamily`]: what it reads from its
//! folder, how its model loads, how it admits and answers a request, and how
//! it trains its whole model. Laya, Kev and Clef each carried their own copy
//! of this shell before; a fix to it now reaches all three.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use decision_contract::decision_training_job::{DecisionTrainingJob, TrainingExample};
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{
    DecisionFailure, DecisionQuestion, HeadInputRequest, OrderedJsonValue, QuestionOutcome,
};
use decision_training::head_parameters::ParameterLayout;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::decision_engine::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, DecisionTrainingStep,
};
#[cfg(feature = "gradient-check")]
use crate::decision_models::training::directional_gradient_check::{
    ArrayGradientCheck, GradientCheckProgress,
};
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::{
    DecisionTrainingHost, DecisionTrainingSession, WholeModelScorer, keep_session_if_running,
    read_through_decide, start_session,
};
use crate::memory::telemetry::{MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
use memory_policy::live_memory_limit::{LiveMemoryLimitEngine, LiveMemoryLimitRefusal};
use model_primitives::forward_pass_memory::forward_pass_memory_telemetry;
use model_primitives::live_memory_ceiling::move_engine_ceiling;

/// What a decision family adds to the shared shell.
pub trait DecisionFamily: 'static {
    /// What the family reads and checks from its folder at construction, so
    /// an unusable folder is refused before any weights load.
    type Prepared: 'static;
    /// The loaded model. It is shared, because a whole-model training's loss
    /// is built inside MLX closures that must own everything they read.
    type LoadedState: 'static;
    /// What admitting a request prepares for answering it, such as the
    /// token sequences a family builds before any model work.
    type RequestPlan;
    /// Which trained-scorer files this family reads.
    const TUNABLE_FAMILY: TunableHeadFamily;

    fn prepare(model_directory: &Path) -> Result<PreparedFolder<Self::Prepared>, DecisionFailure>;

    /// Loads the model. `prepared` is what construction read; a family that
    /// consumes it takes it out, so a second load finds it gone.
    fn load_state(
        model_directory: &Path,
        prepared: &mut Option<Self::Prepared>,
        memory_limits: MlxMemoryLimits,
    ) -> Result<LoadedDecisionModel<Self::LoadedState>, DecisionFailure>;

    fn runtime(loaded_state: &Self::LoadedState) -> &MlxRuntime;

    /// The same runtime, mutably, so a live memory-ceiling change can reach it.
    fn runtime_mut(loaded_state: &mut Self::LoadedState) -> &mut MlxRuntime;

    /// Checks a request before any model work, in the family's own order and
    /// wording, and returns the loaded model to answer it with and its plan.
    /// A refusal here leaves the memory report untouched, as no model work ran.
    fn admit_decision<'loaded>(
        loaded_state: Option<&'loaded Rc<Self::LoadedState>>,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<AdmittedDecision<'loaded, Self::LoadedState, Self::RequestPlan>, DecisionFailure>;

    /// Every question's answer and the tokens the model read.
    fn answer_questions(
        loaded_state: &Rc<Self::LoadedState>,
        request_plan: Self::RequestPlan,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure>;

    fn whole_model_scorer(
        loaded_state: &Rc<Self::LoadedState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure>;
}

/// A request that passed the family's checks, with what it needs to be answered.
pub struct AdmittedDecision<'loaded, LoadedState, RequestPlan> {
    pub loaded_state: &'loaded Rc<LoadedState>,
    pub request_plan: RequestPlan,
}

/// What construction read from a folder.
pub struct PreparedFolder<Prepared> {
    pub prepared: Prepared,
    /// The weights the model keeps resident, which memory reports count.
    pub resident_weight_bytes: u64,
}

/// What a family's load produced.
pub struct LoadedDecisionModel<LoadedState> {
    pub loaded_state: LoadedState,
    pub maximum_tokens_per_question: u32,
}

/// A decision runtime bound to one model folder.
pub struct DecisionEngineShell<Family: DecisionFamily> {
    model_id: String,
    model_directory: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    resident_weight_bytes: u64,
    prepared: Option<Family::Prepared>,
    loaded_state: Option<Rc<Family::LoadedState>>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
    /// A training of the final layers in progress, if any.
    training_session: Option<DecisionTrainingSession>,
}

impl<Family: DecisionFamily> DecisionEngineShell<Family> {
    /// Creates an unloaded engine for `model_directory`, refusing a folder
    /// the family cannot use before any weights are loaded.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        let prepared_folder = Family::prepare(model_directory)?;
        Ok(Self {
            model_id: model_id.into(),
            model_directory: model_directory.to_path_buf(),
            effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes,
            resident_weight_bytes: prepared_folder.resident_weight_bytes,
            prepared: Some(prepared_folder.prepared),
            loaded_state: None,
            post_cleanup_memory_telemetry: None,
            training_session: None,
        })
    }

    /// The loaded model, if `load` has run. Public because each family is a
    /// crate of its own, and its layout helpers read the state from there.
    pub fn loaded_state(&self) -> Option<&Rc<Family::LoadedState>> {
        self.loaded_state.as_ref()
    }

    fn memory_telemetry(&self, runtime: &MlxRuntime) -> Option<MlxMemoryTelemetry> {
        let memory_snapshot = runtime.memory_snapshot().ok()?;
        Some(forward_pass_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes as u64,
            self.resident_weight_bytes,
        ))
    }

    #[cfg(feature = "gradient-check")]
    /// Checks, without training, that `job`'s gradient at the starting
    /// parameters is the loss's own slope for every trainable array, over
    /// the batch of its first `checked_row_count` training rows. A backend
    /// whose backward pass is wrong passes the parity check yet fails this
    /// one (issue #2699).
    ///
    /// # Errors
    ///
    /// Returns the failure of loading, reading, preparing or any pass.
    pub fn check_starting_gradient(
        &mut self,
        job: DecisionTrainingJob,
        checked_row_count: usize,
        progress: &mut dyn GradientCheckProgress,
    ) -> Result<Vec<ArrayGradientCheck>, DecisionFailure> {
        self.loaded_for_training()?;
        let mut session = DecisionTrainingSession::new(job)?;
        session.check_starting_gradient(self, checked_row_count, progress)
    }

    fn loaded_for_training(&self) -> Result<&Rc<Family::LoadedState>, DecisionFailure> {
        match self.loaded_state.as_ref() {
            Some(loaded_state) => Ok(loaded_state),
            None => Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to train before load",
            )),
        }
    }
}

impl<Family: DecisionFamily> DecisionEngine for DecisionEngineShell<Family> {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        if self.loaded_state.is_some() {
            return Err(DecisionFailure::fatal_execution(
                "decision engine load was requested twice",
            ));
        }
        let Ok(memory_limits) = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.mlx_buffer_cache_limit_bytes,
        ) else {
            return Err(DecisionFailure::fatal_execution(
                "decision runtime memory limits are invalid",
            ));
        };
        let loaded_model =
            Family::load_state(&self.model_directory, &mut self.prepared, memory_limits)?;
        self.loaded_state = Some(Rc::new(loaded_model.loaded_state));
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question: loaded_model.maximum_tokens_per_question,
        })
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        let started_at = Instant::now();
        let admitted_decision =
            Family::admit_decision(self.loaded_state.as_ref(), state, questions)?;
        let loaded_state = admitted_decision.loaded_state;
        let decision_outcome = Family::answer_questions(
            loaded_state,
            admitted_decision.request_plan,
            state,
            questions,
            head_input_request,
        );
        // Whatever happened, the request's arrays are dropped by now;
        // clearing the mlx_buffer_cache returns their memory before the next
        // request, and the report taken after it is what that request starts from.
        let runtime = Family::runtime(loaded_state);
        let _cleanup_result = runtime.synchronize_gpu_stream_and_clear_mlx_buffer_cache();
        self.post_cleanup_memory_telemetry = self.memory_telemetry(runtime);
        let (question_outcomes, input_token_count) = decision_outcome?;
        Ok(DecisionEngineOutput {
            question_outcomes,
            input_token_count,
            elapsed_millis: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn start_training(&mut self, job: DecisionTrainingJob) -> Result<(), DecisionFailure> {
        self.loaded_for_training()?;
        start_session(&mut self.training_session, job)
    }

    fn advance_training(&mut self) -> Result<DecisionTrainingStep, DecisionFailure> {
        let Some(mut session) = self.training_session.take() else {
            return Err(DecisionFailure::fatal_execution(
                "no training is in progress",
            ));
        };
        let step = session.advance(self);
        keep_session_if_running(&mut self.training_session, session, &step);
        step
    }

    fn cancel_training(&mut self) {
        self.training_session = None;
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let loaded_state = self.loaded_state.as_ref()?;
        self.memory_telemetry(Family::runtime(loaded_state))
    }
}

impl<Family: DecisionFamily> LiveMemoryLimitEngine for DecisionEngineShell<Family> {
    fn update_mlx_memory_limit(
        &mut self,
        requested_mlx_memory_ceiling_bytes: u64,
    ) -> Result<MlxMemoryLimitAdjustment, LiveMemoryLimitRefusal> {
        // A training run reads the model across steps and sized its work
        // against the ceiling it started under.
        if self.training_session.is_some() {
            return Err(LiveMemoryLimitRefusal::EngineBusy);
        }
        // A shared handle means something else still reads the model, such
        // as a whole-model scorer built for a training; it must not see the
        // runtime change under it.
        let loaded_runtime = match self.loaded_state.as_mut() {
            Some(loaded_state) => match Rc::get_mut(loaded_state) {
                Some(unshared_state) => Some(Family::runtime_mut(unshared_state)),
                None => return Err(LiveMemoryLimitRefusal::EngineBusy),
            },
            None => None,
        };
        let applied_ceiling = move_engine_ceiling(
            loaded_runtime,
            &mut self.effective_mlx_memory_ceiling_bytes,
            &mut self.mlx_buffer_cache_limit_bytes,
            requested_mlx_memory_ceiling_bytes,
        )?;
        Ok(applied_ceiling.into_adjustment(
            requested_mlx_memory_ceiling_bytes,
            self.collect_mlx_memory_telemetry(),
        ))
    }
}

impl<Family: DecisionFamily> DecisionTrainingHost for DecisionEngineShell<Family> {
    fn read_answers(
        &mut self,
        state: &OrderedJsonValue,
        questions: &OrderedJsonValue,
        head_input_request: HeadInputRequest,
    ) -> Result<Vec<QuestionOutcome>, DecisionFailure> {
        read_through_decide(self, state, questions, head_input_request)
    }

    fn mlx_runtime(&self) -> Result<&MlxRuntime, DecisionFailure> {
        Ok(Family::runtime(self.loaded_for_training()?))
    }

    fn model_directory(&self) -> &Path {
        &self.model_directory
    }

    fn tunable_family(&self) -> TunableHeadFamily {
        Family::TUNABLE_FAMILY
    }

    fn whole_model_scorer(
        &mut self,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        Family::whole_model_scorer(
            self.loaded_for_training()?,
            examples,
            row_sources,
            scorer_layout,
        )
    }
}

/// The checks Kev and Clef make before answering: a request needs a
/// question, then a loaded model, worded with the family's name.
pub fn admit_questions_then_load<'loaded, LoadedState>(
    family_name: &str,
    loaded_state: Option<&'loaded Rc<LoadedState>>,
    questions: &[DecisionQuestion],
) -> Result<AdmittedDecision<'loaded, LoadedState, ()>, DecisionFailure> {
    if questions.is_empty() {
        return Err(DecisionFailure::invalid_request(
            "a decision request needs at least one question",
        ));
    }
    match loaded_state {
        Some(loaded_state) => Ok(AdmittedDecision {
            loaded_state,
            request_plan: (),
        }),
        None => Err(DecisionFailure::fatal_execution(&format!(
            "the {family_name} engine was asked to decide before load"
        ))),
    }
}

/// The bytes of every safetensors file directly in `model_directory`; zero
/// when the folder cannot be read, since the folder is checked elsewhere.
pub fn safetensors_bytes_in(model_directory: &Path) -> u64 {
    let Ok(directory_entries) = std::fs::read_dir(model_directory) else {
        return 0;
    };
    let mut total_bytes = 0;
    for directory_entry in directory_entries.flatten() {
        let file_name = directory_entry.file_name();
        if file_name.to_string_lossy().ends_with(".safetensors")
            && let Ok(file_metadata) = directory_entry.metadata()
        {
            total_bytes += file_metadata.len();
        }
    }
    total_bytes
}

#[cfg(test)]
#[path = "decision_engine_shell_tests.rs"]
mod decision_engine_shell_tests;
