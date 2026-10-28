//! Explicit test inputs, exact Cargo executables, and retained per-run evidence.

mod arguments;
mod cargo_artifacts;
mod process;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use astronomical_test_harness::test_run::{TestRunConfiguration, TestRunContext};

use crate::process_signals::SignalPipe;
use arguments::{TestRunRequest, TestTarget};
use cargo_artifacts::{
    listed_test_names, parse_artifacts, test_build_arguments, workspace_binary_build_arguments,
};
use process::{CommandOutput, OutputStyle, run_command};

const BUILD_BUDGET: Duration = Duration::from_secs(600);
const TEST_BUDGET: Duration = Duration::from_secs(120);
const LISTING_BUDGET: Duration = Duration::from_secs(15);

pub(crate) fn run(repository_root: &Path, arguments: &[String]) -> ExitCode {
    let request = match TestRunRequest::parse(arguments) {
        Ok(request) => request,
        Err(usage_error) => {
            eprintln!("{usage_error}\nusage: {}", arguments::USAGE);
            return ExitCode::from(2);
        }
    };
    let configuration = match load_configuration(request.configuration_path.as_deref()) {
        Ok(configuration) => configuration,
        Err(configuration_error) => {
            eprintln!("{configuration_error}");
            return ExitCode::FAILURE;
        }
    };
    finish_run(execute(repository_root, &request, configuration))
}

pub(crate) fn refresh_coverage_inventory(repository_root: &Path, arguments: &[String]) -> ExitCode {
    if !arguments.is_empty() {
        eprintln!("cargo xtask mlx-c-coverage-inventory does not accept arguments");
        return ExitCode::from(2);
    }
    let started_at = Instant::now();
    let inventory_path = repository_root.join("crates/mlx-c-fork/COVERAGE_INVENTORY.md");
    eprintln!(
        "[mlx-c-coverage-inventory] status=start timeout_seconds={}",
        BUILD_BUDGET.as_secs()
    );
    let request = TestRunRequest {
        package: "astronomical-runtime-integration".to_owned(),
        target: TestTarget::Integration("mlx_c_coverage_contract_tests".to_owned()),
        features: Some("mlx".to_owned()),
        profile: None,
        configuration_path: None,
        filter: Some("should_keep_the_committed_coverage_inventory_current".to_owned()),
        exact: false,
        ignored: false,
    };
    let configuration = TestRunConfiguration {
        coverage_inventory_output: Some(inventory_path.clone()),
        ..TestRunConfiguration::default()
    };
    match execute(repository_root, &request, configuration) {
        Ok(()) => {
            eprintln!(
                "[mlx-c-coverage-inventory] status=success artifact={} elapsed_seconds={}",
                inventory_path.display(),
                started_at.elapsed().as_secs()
            );
            ExitCode::SUCCESS
        }
        Err(run_failure) => finish_run(Err(run_failure)),
    }
}

fn finish_run(run_outcome: Result<(), RunFailure>) -> ExitCode {
    match run_outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(run_failure) => {
            eprintln!("{}", run_failure.message);
            ExitCode::from(run_failure.exit_code)
        }
    }
}

#[expect(clippy::too_many_lines)]
fn execute(
    repository_root: &Path,
    request: &TestRunRequest,
    mut configuration: TestRunConfiguration,
) -> Result<(), RunFailure> {
    configuration.validate()?;
    let repository_root = repository_root
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let run_directory = create_run_directory(&repository_root)?;
    eprintln!(
        "[test-run] status=prepared retained_run_directory={}",
        run_directory.display()
    );
    if configuration.evidence_directory.is_none() {
        configuration.evidence_directory = Some(run_directory.join("evidence"));
    } else if let Some(evidence_root) = &configuration.evidence_directory {
        configuration.evidence_directory = Some(
            evidence_root.join(
                run_directory
                    .file_name()
                    .ok_or_else(|| "the owned run directory has no name".to_owned())?,
            ),
        );
    }
    let signals = SignalPipe::install()
        .ok_or_else(|| "could not install test-run interruption handlers".to_owned())?;
    let source_commit = git_text(&repository_root, &["rev-parse", "HEAD"])?;
    let mut run_context = TestRunContext {
        configuration,
        source_commit: Some(source_commit.clone()),
        source_is_dirty: !git_text(&repository_root, &["status", "--porcelain"])?.is_empty(),
        ..TestRunContext::default()
    };

    let built_tests = cargo_command(
        &repository_root,
        &run_directory,
        "build-tests",
        &test_build_arguments(request),
        &signals,
    )?;
    let mut artifacts = parse_artifacts(&built_tests.stdout, &repository_root)?;
    if let Some(binary_arguments) = workspace_binary_build_arguments(request) {
        let built_binaries = cargo_command(
            &repository_root,
            &run_directory,
            "build-workspace-binaries",
            &binary_arguments,
            &signals,
        )?;
        let workspace_artifacts = parse_artifacts(&built_binaries.stdout, &repository_root)?;
        artifacts.binaries.extend(workspace_artifacts.binaries);
    }
    if git_text(&repository_root, &["rev-parse", "HEAD"])? != source_commit {
        return Err(
            "source commit changed during the test build; run again against one revision"
                .to_owned()
                .into(),
        );
    }
    run_context.source_is_dirty |=
        !git_text(&repository_root, &["status", "--porcelain"])?.is_empty();
    run_context.workspace_binaries = artifacts.binaries;
    run_context.workspace_binary_sha256 =
        cargo_artifacts::binary_digests(&run_context.workspace_binaries)?;
    run_context
        .write_to_directory(&run_directory)
        .map_err(|error| error.to_string())?;
    if artifacts.tests.is_empty() {
        return Err("Cargo reported no test executable for the selected target"
            .to_owned()
            .into());
    }
    let mut selected_cases = Vec::new();
    for (executable_index, test_executable) in artifacts.tests.iter().enumerate() {
        let selected_names = select_test_names(
            test_executable,
            executable_index,
            request,
            &run_directory,
            &signals,
        )?;
        selected_cases.extend(
            selected_names
                .into_iter()
                .map(|test_name| (test_executable.clone(), test_name)),
        );
    }
    if selected_cases.is_empty() {
        return Err("the explicit test selection matched no runnable tests"
            .to_owned()
            .into());
    }
    let cases_started_at = Instant::now();
    for (case_index, (test_executable, test_name)) in selected_cases.iter().enumerate() {
        let case_directory = run_directory.join(format!("case-{case_index}"));
        fs::create_dir(&case_directory).map_err(|error| error.to_string())?;
        let mut case_context = run_context.clone();
        case_context.configuration.evidence_directory = run_context
            .configuration
            .evidence_directory
            .as_ref()
            .map(|evidence_root| evidence_root.join(format!("case-{case_index}")));
        case_context
            .write_to_directory(&case_directory)
            .map_err(|error| error.to_string())?;
        let mut test_command = Command::new(test_executable);
        test_command.current_dir(&case_directory).args([
            test_name,
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ]);
        if request.ignored {
            test_command.arg("--ignored");
        }
        let test_output = run_command(
            &mut test_command,
            &case_directory.join("test.log"),
            test_name,
            TEST_BUDGET,
            OutputStyle::Test,
            &signals,
        )?;
        require_success(test_output, test_name)?;
        let completed_cases = case_index + 1;
        let remaining_cases = selected_cases.len() - completed_cases;
        let estimated_remaining_seconds = cases_started_at.elapsed().as_secs_f64()
            / completed_cases as f64
            * remaining_cases as f64;
        eprintln!(
            "[test-run] status=case-finished completed={completed_cases} total={} progress_percent={:.1} eta_seconds={estimated_remaining_seconds:.1}",
            selected_cases.len(),
            100.0 * completed_cases as f64 / selected_cases.len() as f64
        );
    }
    Ok(())
}

fn cargo_command(
    repository_root: &Path,
    run_directory: &Path,
    stage: &str,
    cargo_arguments: &[String],
    signals: &SignalPipe,
) -> Result<CommandOutput, RunFailure> {
    let mut cargo_command = Command::new("cargo");
    cargo_command
        .current_dir(repository_root)
        .args(cargo_arguments);
    let command_output = run_command(
        &mut cargo_command,
        &run_directory.join(format!("{stage}.log")),
        stage,
        BUILD_BUDGET,
        OutputStyle::Cargo,
        signals,
    )?;
    require_success(command_output, stage)
}

fn select_test_names(
    test_executable: &Path,
    executable_index: usize,
    request: &TestRunRequest,
    run_directory: &Path,
    signals: &SignalPipe,
) -> Result<Vec<String>, RunFailure> {
    let read_listing = |ignored_only| {
        let stage = format!("list-{executable_index}-ignored-{ignored_only}");
        let mut listing_command = Command::new(test_executable);
        listing_command
            .current_dir(run_directory)
            .args(request.listing_arguments(ignored_only));
        let listing = run_command(
            &mut listing_command,
            &run_directory.join(format!("{stage}.log")),
            &stage,
            LISTING_BUDGET,
            OutputStyle::Quiet,
            signals,
        )?;
        require_success(listing, &stage)
    };
    if request.ignored {
        return Ok(listed_test_names(&read_listing(true)?.stdout));
    }
    let test_names = listed_test_names(&read_listing(false)?.stdout);
    let ignored_names: BTreeSet<_> = listed_test_names(&read_listing(true)?.stdout)
        .into_iter()
        .collect();
    Ok(test_names
        .into_iter()
        .filter(|test_name| !ignored_names.contains(test_name))
        .collect())
}

fn load_configuration(configuration_path: Option<&Path>) -> Result<TestRunConfiguration, String> {
    let Some(configuration_path) = configuration_path else {
        return Ok(TestRunConfiguration::default());
    };
    let configuration_path = configuration_path
        .canonicalize()
        .map_err(|error| format!("cannot resolve test configuration: {error}"))?;
    let configuration_bytes = fs::read(&configuration_path)
        .map_err(|error| format!("cannot read test configuration: {error}"))?;
    let mut configuration: TestRunConfiguration = serde_json::from_slice(&configuration_bytes)
        .map_err(|error| format!("invalid test configuration: {error}"))?;
    configuration.validate()?;
    configuration.resolve_paths(
        configuration_path
            .parent()
            .ok_or_else(|| "the test configuration has no parent directory".to_owned())?,
    );
    if configuration
        .model_directory
        .as_ref()
        .is_some_and(|model_directory| !model_directory.is_dir())
    {
        return Err("the explicitly selected model_directory does not exist".to_owned());
    }
    Ok(configuration)
}

fn create_run_directory(repository_root: &Path) -> Result<PathBuf, String> {
    let runs_directory = repository_root.join("target/test-runs");
    fs::create_dir_all(&runs_directory).map_err(|error| error.to_string())?;
    let timestamp_nanoseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let run_directory =
        runs_directory.join(format!("{}-{timestamp_nanoseconds}", std::process::id()));
    fs::create_dir(&run_directory).map_err(|error| error.to_string())?;
    Ok(run_directory)
}

fn git_text(repository_root: &Path, arguments: &[&str]) -> Result<String, String> {
    let git_output = Command::new("git")
        .arg("--no-pager")
        .args(arguments)
        .current_dir(repository_root)
        .output()
        .map_err(|error| format!("cannot read source identity: {error}"))?;
    if !git_output.status.success() {
        return Err(format!(
            "cannot read source identity: {}",
            String::from_utf8_lossy(&git_output.stderr)
        ));
    }
    String::from_utf8(git_output.stdout)
        .map(|source_identity| source_identity.trim().to_owned())
        .map_err(|error| format!("invalid source identity: {error}"))
}

struct RunFailure {
    exit_code: u8,
    message: String,
}

impl From<String> for RunFailure {
    fn from(message: String) -> Self {
        Self {
            exit_code: 1,
            message,
        }
    }
}

fn require_success(
    command_output: CommandOutput,
    stage: &str,
) -> Result<CommandOutput, RunFailure> {
    if command_output.exit_code == 0 {
        return Ok(command_output);
    }
    Err(RunFailure {
        exit_code: u8::try_from(command_output.exit_code).unwrap_or(1),
        message: format!(
            "{stage} failed with status {}; the run directory and evidence were retained",
            command_output.exit_code
        ),
    })
}

#[cfg(test)]
#[path = "test_run_tests.rs"]
mod test_run_tests;
