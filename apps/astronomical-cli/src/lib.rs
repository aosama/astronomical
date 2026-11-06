//! User-facing `astronomical` CLI. Launch prepares a coding harness against a
//! running loopback Astronomical instance; it is not the daemon.

#![forbid(unsafe_code)]

pub mod arguments;
/// The wait `respond` and `embed` share, kept outside both so neither verb owns it.
mod daemon_frame_wait;
pub mod formatting;
pub mod instance;
pub mod model_lifecycle;
pub mod transport;
pub mod verbs;

pub use arguments::{CliCommand, LaunchArguments, UsageError, help_text, parse_command};
pub use instance::{candidate_instances, runtime_instance_from_executable_path};
pub use model_lifecycle::{ModelLifecycle, RequiredCapability};
pub use transport::ipc::{DaemonProbe, DaemonProbeError, DaemonStatusSnapshot};
pub use verbs::doctor::{DoctorDependencies, DoctorError, run_doctor};
pub use verbs::embed::{EmbedArguments, EmbedDependencies, EmbedError, run_embed};
pub use verbs::image::{ImageArguments, ImageDependencies, ImageError, run_image};
pub use verbs::launch::{LaunchDependencies, LaunchError, PreparedLaunch, prepare_launch};
pub use verbs::model_programs::{ModelProgramsReport, render_model_programs};
pub use verbs::models::{ModelsCommand, ModelsDependencies, ModelsError, run_models};
pub use verbs::quantize::{
    QuantizeCommand, QuantizeConvertArguments, QuantizeDependencies, QuantizeError,
    QuantizeInspectArguments, QuantizePlanArguments, run_quantize_convert, run_quantize_inspect,
    run_quantize_plan,
};
pub use verbs::respond::{RespondArguments, RespondDependencies, RespondError, run_respond};
pub use verbs::schema::{
    SchemaArguments, SchemaPropertyInput, SchemaPropertyKind, build_schema_document, run_schema,
};
pub use verbs::status::{StatusDependencies, StatusError, run_status};
pub use verbs::validate_config::{
    ValidateConfigArguments, ValidateConfigDependencies, ValidateConfigError, run_validate_config,
};
pub use verbs::vision::{
    VisionAddArguments, VisionCommand, VisionDependencies, VisionError, VisionInspectArguments,
    VisionMatchArguments, run_vision_add, run_vision_inspect, run_vision_match,
};
