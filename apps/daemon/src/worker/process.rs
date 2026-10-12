//! Worker process lifecycle: launch, containment, replacement, and model swap.

pub(crate) mod worker_containment;
pub(crate) mod worker_model_swap;
pub(crate) mod worker_process;
pub(crate) mod worker_replacement;
pub(crate) mod worker_startup_runtime;
pub(crate) mod worker_stderr_tail;
