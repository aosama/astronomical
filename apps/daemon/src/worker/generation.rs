//! Generation lifecycle: admission, preparation, output collection, and events.

pub(crate) mod admission;
pub(crate) mod completion_event;
pub(crate) mod generate;
pub(crate) mod output;
pub(crate) mod prefill_progress;
pub(crate) mod preparation;

#[cfg(test)]
mod output_tests;
