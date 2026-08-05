use std::{
    env,
    ffi::OsString,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    process::ExitCode,
};

use astronomical_experimental_aligned_expert_packs::{
    AlignedExpertPackPreparationInspection, AlignedExpertPackPreparer,
};

const HELP: &str = "astronomical-experimental-aligned-expert-pack-preparer

Explicitly prepare experimental aligned expert packs for one already-downloaded model.

USAGE:
    astronomical-experimental-aligned-expert-pack-preparer --model-directory PATH [--dry-run] [--yes] [--replace]

OPTIONS:
    --model-directory PATH  Complete downloaded Qwen3.5-MoE model directory
    --dry-run               Validate and report planned disk use without mutation
    --yes                   Confirm preparation without an interactive prompt
    --replace               Replace an invalid existing generated pack revision
    -h, --help              Show this help
    --version               Show the application version
";

#[derive(Debug)]
struct CommandArguments {
    model_directory: PathBuf,
    should_dry_run: bool,
    has_noninteractive_consent: bool,
    should_replace_existing_revision: bool,
}

fn main() -> ExitCode {
    match run(env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(command_error) => {
            eprintln!("astronomical-experimental-aligned-expert-pack-preparer: {command_error}");
            ExitCode::from(command_error.exit_code())
        }
    }
}

fn run(arguments: impl Iterator<Item = OsString>) -> Result<(), CommandError> {
    let Some(command_arguments) = parse_arguments(arguments)? else {
        return Ok(());
    };
    let preparer =
        AlignedExpertPackPreparer::for_model_directory(&command_arguments.model_directory)
            .map_err(CommandError::Preparation)?;
    let preparation_inspection = preparer.inspect().map_err(CommandError::Preparation)?;
    print_inspection(&preparation_inspection);
    if command_arguments.should_dry_run {
        println!(
            "status=dry_run_success model={} revision={} layers={} planned_bytes={} destination={}",
            preparation_inspection.model_id,
            preparation_inspection.model_revision,
            preparation_inspection.total_layer_count,
            preparation_inspection.total_pack_byte_count,
            preparation_inspection.final_revision_directory.display(),
        );
        return Ok(());
    }
    if preparation_inspection.remaining_pack_byte_count
        > preparation_inspection.available_byte_count
    {
        return Err(CommandError::InsufficientAvailableSpace {
            required_byte_count: preparation_inspection.remaining_pack_byte_count,
            available_byte_count: preparation_inspection.available_byte_count,
        });
    }
    if !preparation_inspection.has_valid_final_revision
        && !command_arguments.has_noninteractive_consent
    {
        request_interactive_consent()?;
    }
    let preparation_report = preparer
        .prepare(
            command_arguments.should_replace_existing_revision,
            |progress_event| {
                let elapsed_seconds = progress_event.elapsed.as_secs_f64();
                let average_seconds_per_layer =
                    elapsed_seconds / progress_event.completed_layer_count as f64;
                let remaining_layer_count = progress_event
                    .total_layer_count
                    .saturating_sub(progress_event.completed_layer_count);
                eprintln!(
                    "status=progress completed_layers={}/{} layer_index={} layer_bytes={} total_completed_bytes={} elapsed_seconds={:.2} ETA_seconds={:.1}",
                    progress_event.completed_layer_count,
                    progress_event.total_layer_count,
                    progress_event.layer_index,
                    progress_event.layer_byte_count,
                    progress_event.total_completed_byte_count,
                    elapsed_seconds,
                    average_seconds_per_layer * remaining_layer_count as f64,
                );
            },
        )
        .map_err(CommandError::Preparation)?;
    println!(
        "status=success model={} revision={} layers={} total_bytes={} reused_existing_pack_set={} elapsed_seconds={:.2} destination={}",
        preparation_report.model_id,
        preparation_report.model_revision,
        preparation_report.completed_layer_count,
        preparation_report.total_pack_byte_count,
        preparation_report.reused_existing_pack_set,
        preparation_report.elapsed.as_secs_f64(),
        preparation_report.final_revision_directory.display(),
    );
    eprintln!("Prepared packs remain experimental and are ignored by Astronomical production.");
    Ok(())
}

fn parse_arguments(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<Option<CommandArguments>, CommandError> {
    let mut model_directory = None;
    let mut should_dry_run = false;
    let mut has_noninteractive_consent = false;
    let mut should_replace_existing_revision = false;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("-h" | "--help") => {
                print!("{HELP}");
                return Ok(None);
            }
            Some("--version") => {
                println!(env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            Some("--model-directory") => {
                if model_directory.is_some() {
                    return Err(CommandError::Usage(
                        "--model-directory may be supplied only once".to_owned(),
                    ));
                }
                model_directory = Some(PathBuf::from(arguments.next().ok_or_else(|| {
                    CommandError::Usage("--model-directory requires PATH".to_owned())
                })?));
            }
            Some("--dry-run") if !should_dry_run => should_dry_run = true,
            Some("--yes") if !has_noninteractive_consent => has_noninteractive_consent = true,
            Some("--replace") if !should_replace_existing_revision => {
                should_replace_existing_revision = true;
            }
            Some(known_repeated_flag @ ("--dry-run" | "--yes" | "--replace")) => {
                return Err(CommandError::Usage(format!(
                    "{known_repeated_flag} may be supplied only once"
                )));
            }
            Some(unknown_argument) => {
                return Err(CommandError::Usage(format!(
                    "unknown argument {unknown_argument:?}; use --help"
                )));
            }
            None => {
                return Err(CommandError::Usage(
                    "arguments must use valid UTF-8".to_owned(),
                ));
            }
        }
    }
    let model_directory = model_directory.ok_or_else(|| {
        CommandError::Usage("--model-directory PATH is required; use --help".to_owned())
    })?;
    if should_dry_run && has_noninteractive_consent {
        return Err(CommandError::Usage(
            "--yes cannot be combined with --dry-run".to_owned(),
        ));
    }
    if should_dry_run && should_replace_existing_revision {
        return Err(CommandError::Usage(
            "--replace cannot be combined with --dry-run".to_owned(),
        ));
    }
    Ok(Some(CommandArguments {
        model_directory,
        should_dry_run,
        has_noninteractive_consent,
        should_replace_existing_revision,
    }))
}

fn print_inspection(preparation_inspection: &AlignedExpertPackPreparationInspection) {
    eprintln!(
        "status=preflight model={} revision={} layers={} total_bytes={} total_gb={:.2} remaining_bytes={} remaining_gb={:.2} available_bytes={} available_gb={:.2} valid_final_revision={} destination={}",
        preparation_inspection.model_id,
        preparation_inspection.model_revision,
        preparation_inspection.total_layer_count,
        preparation_inspection.total_pack_byte_count,
        decimal_gigabytes(preparation_inspection.total_pack_byte_count),
        preparation_inspection.remaining_pack_byte_count,
        decimal_gigabytes(preparation_inspection.remaining_pack_byte_count),
        preparation_inspection.available_byte_count,
        decimal_gigabytes(preparation_inspection.available_byte_count),
        preparation_inspection.has_valid_final_revision,
        preparation_inspection.final_revision_directory.display(),
    );
}

fn request_interactive_consent() -> Result<(), CommandError> {
    if !io::stdin().is_terminal() {
        return Err(CommandError::ConsentRequired);
    }
    eprint!("Prepare and publish these generated packs? [y/N] ");
    io::stderr().flush().map_err(CommandError::Io)?;
    let mut confirmation = String::new();
    io::stdin()
        .read_line(&mut confirmation)
        .map_err(CommandError::Io)?;
    if !matches!(confirmation.trim(), "y" | "Y" | "yes" | "YES") {
        return Err(CommandError::ConsentDeclined);
    }
    Ok(())
}

fn decimal_gigabytes(byte_count: u64) -> f64 {
    byte_count as f64 / 1_000_000_000.0
}

#[derive(Debug)]
enum CommandError {
    Usage(String),
    ConsentRequired,
    ConsentDeclined,
    InsufficientAvailableSpace {
        required_byte_count: u64,
        available_byte_count: u64,
    },
    Io(io::Error),
    Preparation(astronomical_experimental_aligned_expert_packs::AlignedExpertPackPreparationError),
}

impl CommandError {
    const fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            _ => 1,
        }
    }
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(description) => formatter.write_str(description),
            Self::ConsentRequired => formatter
                .write_str("explicit consent is required; rerun interactively or supply --yes"),
            Self::ConsentDeclined => formatter.write_str("preparation cancelled without changes"),
            Self::InsufficientAvailableSpace {
                required_byte_count,
                available_byte_count,
            } => write!(
                formatter,
                "insufficient destination space: required {required_byte_count} bytes, available {available_byte_count} bytes"
            ),
            Self::Io(source) => write!(formatter, "input/output failed: {source}"),
            Self::Preparation(source) => source.fmt(formatter),
        }
    }
}
