//! The memory ledger: one owner for the whole MLX limit (issue #2221).
//!
//! Every byte under the limit belongs to exactly one named account. The
//! accounts that have to exist come first: the weights that never stream,
//! the conversation's stored state, one layer's working memory, the slots a
//! streamed layer is read into, any short-lived leases, and a safety margin.
//! Whatever is left is the capacity for kept weights. Nothing sets that
//! capacity directly; it is always the remainder, so it can only change when
//! one of the accounts in front of it changes.
//!
//! The ledger is arithmetic over numbers the engine hands it. It samples no
//! runtime and keeps no weights, which is why a whole request's memory
//! behaviour can be replayed in a test without a model.
//!
//! - `phase` — what the model is doing, as far as memory is concerned.
//! - `model_shape` — what a loaded model occupies and could keep.
//! - `working_memory` — the measured working memory of one layer, by phase.
//! - `lease` — short-lived needs outside the normal pattern.
//! - `accounts` — the accounts at one instant and the rule that composes them.
//! - `memory_ledger` — the owner: the events that change the accounts.
//! - `reconciliation` — the ledger compared with what MLX reports.

pub mod accounts;
pub mod lease;
pub mod memory_ledger;
pub mod model_shape;
pub mod phase;
pub mod reconciliation;
pub mod working_memory;

pub use accounts::LedgerAccounts;
pub use lease::{LeaseId, LeasePurpose};
pub use memory_ledger::{MemoryLedger, MemoryLedgerError};
pub use model_shape::ModelMemoryShape;
pub use phase::LedgerPhase;
pub use reconciliation::{HeldMemory, LedgerReconciliation};
pub use working_memory::{WorkingMemoryMeasurements, WorkingMemoryReserve};
