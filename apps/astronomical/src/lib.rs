//! User-facing `astronomical` CLI. Launch prepares a coding harness against a
//! running loopback Astronomical instance; it is not the daemon.

#![forbid(unsafe_code)]

pub mod arguments;
pub mod daemon_probe;
pub mod embed;
pub mod embed_arguments;
pub mod errors;
pub mod formatting;
pub mod http;
pub mod instance;
pub mod launch;
pub mod model_lifecycle;
pub mod models;
pub mod models_arguments;
pub mod models_command;
pub mod opencode;
pub mod prompt;
pub mod respond;
pub mod respond_arguments;
pub mod respond_image;
pub mod schema;
pub mod schema_arguments;
pub mod status_command;
pub mod tools;
pub mod validate_config;
pub mod validate_config_arguments;

pub use arguments::{CliCommand, LaunchArguments, help_text, parse_command};
pub use daemon_probe::{DaemonProbe, DaemonProbeError, DaemonStatusSnapshot};
pub use embed::{EmbedDependencies, run_embed};
pub use embed_arguments::EmbedArguments;
pub use errors::{EmbedError, LaunchError, ModelsError, RespondError, StatusError, UsageError};
pub use instance::{candidate_instances, runtime_instance_from_executable_path};
pub use launch::{LaunchDependencies, PreparedLaunch, prepare_launch};
pub use model_lifecycle::{ModelLifecycle, RequiredCapability};
pub use models_arguments::ModelsCommand;
pub use models_command::{ModelsDependencies, run_models};
pub use respond::{RespondDependencies, run_respond};
pub use respond_arguments::RespondArguments;
pub use schema::{build_schema_document, run_schema};
pub use schema_arguments::{SchemaArguments, SchemaPropertyInput, SchemaPropertyKind};
pub use status_command::{StatusDependencies, run_status};
pub use validate_config::{ValidateConfigDependencies, ValidateConfigError, run_validate_config};
pub use validate_config_arguments::ValidateConfigArguments;
