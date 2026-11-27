//! Carrying a folder's training record into every answer it gives.
//!
//! `astronomical decision` writes whatever the person asked for and records
//! what judging it showed in `decision_training_record.json`. A folder judged
//! worse, or never judged, must not answer as if it had passed, so every
//! family's engine is wrapped here and each answer carries the record.
//! Wrapping keeps the families free of it: they answer exactly as before, and
//! a folder without the file answers exactly as before.
//!
//! A record that cannot be read refuses to load, as a calibration file does:
//! answering without it would hide what the person was told about the model.

use std::path::Path;

use decision_contract::decision_training_job::DecisionTrainingJob;
use decision_contract::training_record::TrainingRecord;
use decision_contract::{DecisionFailure, DecisionQuestion, HeadInputRequest, OrderedJsonValue};
use decision_training::training_record_file::read_training_record_file;

use crate::decision_engine::{
    DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput, DecisionTrainingStep,
};
use crate::memory::telemetry::MlxMemoryTelemetry;

/// The folder's training record, or `None` when the folder has none.
pub fn read_training_record(
    model_directory: &Path,
) -> Result<Option<TrainingRecord>, DecisionFailure> {
    read_training_record_file(model_directory)
        .map_err(|record_error| DecisionFailure::fatal_execution(&record_error.to_string()))
}

/// A family's engine whose answers carry the folder's training record.
pub struct RecordedTrainingEngine {
    family_engine: Box<dyn DecisionEngine>,
    training_record: TrainingRecord,
}

impl RecordedTrainingEngine {
    /// `family_engine` itself when the folder has no record, so a folder
    /// `astronomical decision` never wrote is served exactly as before.
    pub fn wrapping(
        family_engine: Box<dyn DecisionEngine>,
        model_directory: &Path,
    ) -> Result<Box<dyn DecisionEngine>, DecisionFailure> {
        Ok(match read_training_record(model_directory)? {
            Some(training_record) => Box::new(Self {
                family_engine,
                training_record,
            }),
            None => family_engine,
        })
    }
}

impl memory_policy::live_memory_limit::LiveMemoryLimitEngine for RecordedTrainingEngine {}

impl DecisionEngine for RecordedTrainingEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        self.family_engine.load()
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        let mut engine_output = self
            .family_engine
            .decide(state, questions, head_input_request)?;
        for question_outcome in &mut engine_output.question_outcomes {
            question_outcome.training_record = Some(self.training_record.clone());
        }
        Ok(engine_output)
    }

    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.family_engine.minimum_mlx_memory_ceiling_bytes()
    }

    fn start_training(&mut self, job: DecisionTrainingJob) -> Result<(), DecisionFailure> {
        self.family_engine.start_training(job)
    }

    fn advance_training(&mut self) -> Result<DecisionTrainingStep, DecisionFailure> {
        self.family_engine.advance_training()
    }

    fn cancel_training(&mut self) {
        self.family_engine.cancel_training();
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.family_engine.take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.family_engine.collect_mlx_memory_telemetry()
    }
}

#[cfg(test)]
#[path = "recorded_training_tests.rs"]
mod tests;
