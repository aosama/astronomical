//! Decision-model backends: one folder per model family.
//!
//! Every backend implements [`crate::DecisionEngine`] and shares the
//! decision contract crate for questions, calibration and answers. A backend
//! owns only what is particular to its family: how the artifact is laid out,
//! and the forward pass from tokens to one logit per option.

pub(crate) mod clef;
pub(crate) mod kev;
pub(crate) mod laya;
pub(crate) mod qwen_backbone;
