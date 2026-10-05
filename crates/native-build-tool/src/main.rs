//! Thin executable wrapper around the native-build tool library.

use std::process::ExitCode;

use astronomical_native_build_tool::NATIVE_BUILD_TOOL_PREFIX;

fn main() -> ExitCode {
    let command_line_arguments: Vec<String> = std::env::args().skip(1).collect();
    let arguments = match astronomical_native_build_tool::parse_arguments(&command_line_arguments) {
        Ok(arguments) => arguments,
        Err(argument_error) => {
            eprintln!("{NATIVE_BUILD_TOOL_PREFIX} {argument_error}");
            return ExitCode::from(2);
        }
    };
    match astronomical_native_build_tool::run_native_build(&arguments) {
        Ok(native_build_outcome) => {
            let outcome_name = if native_build_outcome.was_built() {
                "built"
            } else {
                "reused"
            };
            println!(
                "{NATIVE_BUILD_TOOL_PREFIX} outcome={outcome_name} elapsed_seconds={:.3}",
                native_build_outcome.elapsed().as_secs_f64()
            );
            ExitCode::SUCCESS
        }
        Err(native_build_error) => {
            eprintln!("{NATIVE_BUILD_TOOL_PREFIX} native build failed: {native_build_error}");
            ExitCode::FAILURE
        }
    }
}
