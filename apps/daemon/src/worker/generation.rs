//! Generation lifecycle: admission, preparation, output collection, and events.

pub(crate) mod worker_completion_event;
pub(crate) mod worker_generate;
pub(crate) mod worker_generation_admission;
pub(crate) mod worker_generation_output;
pub(crate) mod worker_generation_preparation;
pub(crate) mod worker_prefill_progress;

#[cfg(test)]
mod worker_generation_output_tests;
