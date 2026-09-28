//! Arguments for `astronomical validate config`.

use std::ffi::OsString;

use astronomical_config::AstronomicalRuntimeInstance;

use crate::errors::UsageError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidateConfigArguments {
    pub runtime_instance: AstronomicalRuntimeInstance,
    pub render_json: bool,
}

pub fn parse_validate_config_arguments(
    remaining_arguments: &[OsString],
) -> Result<ValidateConfigArguments, UsageError> {
    let mut runtime_instance = AstronomicalRuntimeInstance::Development;
    let mut instance_flag_seen = false;
    let mut render_json = false;

    let mut argument_index = 0;
    while argument_index < remaining_arguments.len() {
        let argument_text = remaining_arguments[argument_index]
            .to_string_lossy()
            .into_owned();
        match argument_text.as_str() {
            "--instance" => {
                if instance_flag_seen {
                    return Err(UsageError::RepeatedArgument("--instance"));
                }
                let raw_instance_name = remaining_arguments
                    .get(argument_index + 1)
                    .and_then(|value| value.to_str())
                    .ok_or(UsageError::MissingValue("--instance"))?;
                runtime_instance = match raw_instance_name {
                    "stable" => AstronomicalRuntimeInstance::Stable,
                    "development" => AstronomicalRuntimeInstance::Development,
                    other => return Err(UsageError::UnknownInstance(other.to_owned())),
                };
                instance_flag_seen = true;
                argument_index += 2;
            }
            "--json" => {
                render_json = true;
                argument_index += 1;
            }
            _ => return Err(UsageError::UnknownArgument(argument_text)),
        }
    }

    Ok(ValidateConfigArguments {
        runtime_instance,
        render_json,
    })
}
