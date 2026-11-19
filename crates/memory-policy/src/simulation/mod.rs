//! A scripted engine that drives the ledger through whole requests.
//!
//! The ledger is arithmetic, so the memory behaviour of a request can be
//! replayed without a model or a GPU: a model is a list of layer sizes, and
//! its true MLX working buffers are a number the scenario chooses, which may
//! differ from what the engine was told. Every real memory incident becomes
//! one named scenario here and runs on every commit (issue #2221).
//!
//! - `simulated_model` — a model as sizes, with its true MLX working buffers.
//! - `simulated_engine` — reads prompts one layer at a time and writes
//!   replies, asking the ledger at every step.
//! - `scenarios` — the incidents, replayed.

pub(crate) mod simulated_engine;
pub(crate) mod simulated_model;

mod scenarios;
