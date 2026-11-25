//! The shell's execution core as both Qwen 3.5 engines hold it: loaded from
//! a validated Qwen 3.5 artifact.

use text_generation::generation_loop::execution_core::TextExecutionCore;

use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;

pub type Qwen3_5ExecutionCore<Model, RequestExtension> =
    TextExecutionCore<Model, RequestExtension, ValidatedQwen3_5Artifact>;
