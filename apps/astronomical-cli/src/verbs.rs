//! One module per user-facing `astronomical` verb. A verb module owns the
//! verb's whole contract: its argument types, its error type, and its
//! `run_*` entry point. The command-line grammar that dispatches into
//! these verbs lives in `crate::arguments`; the journeys that exercise
//! them over the real IPC boundary live in `tests/integration/`.

pub mod doctor;
pub mod embed;
pub mod launch;
pub mod models;
pub mod quantize;
pub mod respond;
pub mod schema;
pub mod status;
pub mod validate_config;
