//! One module per user-facing `astronomical` verb. A verb module owns the
//! verb's whole contract: its argument types, its error type, and its
//! `run_*` entry point. The command-line grammar that dispatches into
//! these verbs lives in `crate::arguments`; the journeys that exercise
//! them over the real IPC boundary live in `tests/integration/`.

pub mod convert_atlas;
pub mod decide;
pub mod doctor;
pub(crate) mod download_progress_line;
pub mod embed;
pub mod image;
pub mod launch;
pub mod model_programs;
pub mod models;
pub mod quantize;
pub(crate) mod reserved_output_file;
pub mod respond;
pub mod schema;
pub mod speak;
pub mod status;
pub mod validate_config;
pub mod vision;
