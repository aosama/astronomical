//! `astronomical decision`: power-user tools for decision models.
//!
//! - `decision calibrate` fits a decision model's temperatures to a person's
//!   own labelled examples and writes a calibrated copy of the model.
//! - `decision train` trains the model's final layers (or, with
//!   `--whole-model`, every layer) on those examples and writes a trained
//!   copy.
//!
//! Both accept any number of examples and write their copy whatever the
//! result; the report and the copy's training record say what judging it
//! showed, and the person decides.
//!
//! The fitting and training live in the `astronomical-decision-training`
//! crate; these verbs read the examples, have the daemon score them, and
//! report.

mod calibrate;
mod calibration_report;
mod check_set;
mod counted_noun;
mod decision_arguments;
mod decision_tool;
mod example_scoring;
mod model_folder_lookup;
mod new_model_visibility;
mod progress_reporting;
mod saved_model_check;
mod train;
mod training_job;
mod training_report;
mod training_request;

pub use calibrate::run_calibrate;
pub use decision_arguments::{DecisionCommand, DecisionToolArguments, parse_decision_command};
pub use decision_tool::{DecisionToolDependencies, DecisionToolError};
pub use train::run_train;
