//! Decision-model backends: one folder per model family.
//!
//! Every backend implements [`crate::DecisionEngine`] and shares the
//! decision contract crate for questions, calibration and answers. A backend
//! owns only what is particular to its family: how the artifact is laid out,
//! and the forward pass from tokens to one logit per option.

pub mod adapter_training;
pub mod calibration_override;
pub mod clef;
pub mod decision_adapters;
pub mod decision_engine_shell;
pub mod forward_pass_planning;
pub mod head_tuning;
pub mod kev;
pub mod laya;
pub mod qwen_backbone;
pub mod recorded_training;
pub mod training;
