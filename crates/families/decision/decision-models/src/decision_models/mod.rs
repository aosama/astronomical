//! What every decision family shares: the engine shell, calibration and its
//! overrides, adapters, head tuning, forward-pass planning, training and the
//! Qwen 3.5 backbone. Each family is a crate of its own beside the shell
//! (issue #2473, Step 7) and owns only what is particular to it: how the
//! artifact is laid out, and the forward pass from tokens to one logit per
//! option.

pub mod adapter_training;
pub mod calibration_override;
pub mod decision_adapters;
pub mod decision_engine_shell;
pub mod forward_pass_planning;
pub mod head_tuning;
pub mod qwen_backbone;
pub mod recorded_training;
pub mod training;
