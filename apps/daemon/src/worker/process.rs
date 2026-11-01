//! Worker process lifecycle: launch, containment, replacement, and model swap.

pub(crate) mod child;
pub(crate) mod containment;
pub(crate) mod model_swap;
pub(crate) mod relaunch_memory;
pub(crate) mod replacement;
pub(crate) mod startup_runtime;
pub(crate) mod stderr_tail;
