//! `scheduler/scheduler_config.json`: the flow-matching constants the schedule is built from.

use serde::Deserialize;

use crate::qwen_image_21::FlowMatchSchedulerParams;

use super::{QwenImage21ConfigError, parse_document, require};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerDocument {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,
    base_image_seq_len: u32,
    base_shift: f64,
    invert_sigmas: bool,
    max_image_seq_len: u32,
    max_shift: f64,
    num_train_timesteps: u32,
    shift: f64,
    shift_terminal: Option<f64>,
    stochastic_sampling: bool,
    time_shift_type: String,
    use_beta_sigmas: bool,
    use_dynamic_shifting: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}

/// Flow-matching scheduler constants read from `scheduler/scheduler_config.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenImage21SchedulerConfig {
    pub num_train_timesteps: u32,
    pub params: FlowMatchSchedulerParams,
}

impl QwenImage21SchedulerConfig {
    pub fn parse(json_bytes: &[u8]) -> Result<Self, QwenImage21ConfigError> {
        const DOCUMENT: &str = "scheduler/scheduler_config.json";
        let document: SchedulerDocument = parse_document(json_bytes, DOCUMENT)?;
        require(
            document.class_name == "FlowMatchEulerDiscreteScheduler",
            DOCUMENT,
            "_class_name",
        )?;
        require(
            document.diffusers_version == "0.37.0.dev0",
            DOCUMENT,
            "_diffusers_version",
        )?;
        require(
            document.base_image_seq_len == 256,
            DOCUMENT,
            "base_image_seq_len",
        )?;
        require(document.base_shift == 0.5, DOCUMENT, "base_shift")?;
        require(!document.invert_sigmas, DOCUMENT, "invert_sigmas")?;
        require(
            document.max_image_seq_len == 8192,
            DOCUMENT,
            "max_image_seq_len",
        )?;
        require(document.max_shift == 0.9, DOCUMENT, "max_shift")?;
        require(
            document.num_train_timesteps == 1000,
            DOCUMENT,
            "num_train_timesteps",
        )?;
        require(document.shift == 1.0, DOCUMENT, "shift")?;
        require(
            document.shift_terminal == Some(0.02),
            DOCUMENT,
            "shift_terminal",
        )?;
        require(
            !document.stochastic_sampling,
            DOCUMENT,
            "stochastic_sampling",
        )?;
        require(
            document.time_shift_type == "exponential",
            DOCUMENT,
            "time_shift_type",
        )?;
        require(!document.use_beta_sigmas, DOCUMENT, "use_beta_sigmas")?;
        require(
            document.use_dynamic_shifting,
            DOCUMENT,
            "use_dynamic_shifting",
        )?;
        require(
            !document.use_exponential_sigmas,
            DOCUMENT,
            "use_exponential_sigmas",
        )?;
        require(!document.use_karras_sigmas, DOCUMENT, "use_karras_sigmas")?;
        Ok(Self {
            num_train_timesteps: document.num_train_timesteps,
            params: FlowMatchSchedulerParams {
                base_seq_len: document.base_image_seq_len as f64,
                max_seq_len: document.max_image_seq_len as f64,
                base_shift: document.base_shift,
                max_shift: document.max_shift,
                shift_terminal: document.shift_terminal.unwrap_or(0.0),
            },
        })
    }
}
