//! User-facing `astronomical` CLI. Launch prepares a coding harness against a
//! running loopback Astronomical instance; it is not the daemon.

#![forbid(unsafe_code)]

pub mod arguments;
pub mod errors;
pub mod http;
pub mod instance;
pub mod launch;
pub mod models;
pub mod opencode;
pub mod prompt;
pub mod tools;

pub use arguments::{CliCommand, LaunchArguments, help_text, parse_command};
pub use errors::{LaunchError, UsageError};
pub use instance::{candidate_instances, runtime_instance_from_executable_path};
pub use launch::{LaunchDependencies, PreparedLaunch, prepare_launch};
