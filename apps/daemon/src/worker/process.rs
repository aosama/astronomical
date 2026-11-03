//! Worker process lifecycle: launch, containment, replacement, and model swap.

pub(crate) mod child;
pub(crate) mod containment;
pub(crate) mod crash_loop_guard;
pub(crate) mod exit_description;
pub(crate) mod failure_relaunch;
pub(crate) mod model_swap;
pub(crate) mod relaunch_memory;
pub(crate) mod replacement;
pub(crate) mod startup_runtime;
pub(crate) mod stderr_tail;
