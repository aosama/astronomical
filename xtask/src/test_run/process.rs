//! Bounded supervision of an owned command and its descendants.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::process_signals::{self, SignalPipe};

const POLL_INTERVAL: Duration = Duration::from_millis(100);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const TERMINATION_GRACE: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug)]
pub(super) enum OutputStyle {
    Cargo,
    Test,
    Quiet,
}

pub(super) struct CommandOutput {
    pub exit_code: i32,
    pub stdout: String,
}

enum StreamEvent {
    Line { stdout: bool, text: String },
    Failed(String),
}

/// Runs `command` in its own process group until it exits, the budget runs
/// out or a signal arrives, then stops whatever remains of that group.
///
/// Output is drained after the group stops as well as during the run, so
/// lines written just before exit still reach the log and the capture.
pub(super) fn run_command(
    command: &mut Command,
    log_path: &Path,
    stage: &str,
    budget: Duration,
    output_style: OutputStyle,
    signals: &SignalPipe,
) -> Result<CommandOutput, String> {
    let command_log = create_command_log(log_path, command)?;
    let mut child = spawn_owned_process_group(command, stage)?;
    let process_group_id = i32::try_from(child.id())
        .map_err(|_| "the child process identifier exceeds the supported range".to_owned())?;
    let (output_readers, mut captured_output) =
        start_output_capture(&mut child, command_log, output_style);
    let mut supervision_clock = SupervisionClock::start(Instant::now(), budget);
    eprintln!(
        "[test-run] stage={stage} status=start progress_percent=0 eta_seconds=unknown budget_seconds={}",
        budget.as_secs()
    );
    let command_outcome = supervise_until_finished(
        &mut child,
        stage,
        signals,
        &mut supervision_clock,
        &mut captured_output,
    );
    stop_owned_process_group(process_group_id, &mut child)?;
    let captured_stdout = finish_output_capture(output_readers, captured_output, stage)?;
    let exit_code = command_outcome?;
    eprintln!(
        "[test-run] stage={stage} status=finished progress_percent=100 elapsed_seconds={} eta_seconds=0 exit_code={exit_code}",
        supervision_clock.started_at.elapsed().as_secs()
    );
    Ok(CommandOutput {
        exit_code,
        stdout: captured_stdout,
    })
}

/// One reader thread per pipe, both feeding one channel, so stdout and
/// stderr lines reach the log in the order they were read.
fn start_output_capture(
    child: &mut Child,
    command_log: File,
    output_style: OutputStyle,
) -> ([thread::JoinHandle<()>; 2], CapturedOutput) {
    let (stream_sender, stream_receiver) = mpsc::channel();
    let output_readers = [
        start_reader(
            child.stdout.take().expect("piped stdout"),
            true,
            stream_sender.clone(),
        ),
        start_reader(
            child.stderr.take().expect("piped stderr"),
            false,
            stream_sender,
        ),
    ];
    let captured_output = CapturedOutput {
        stream_receiver,
        command_log,
        captured_stdout: String::new(),
        output_style,
    };
    (output_readers, captured_output)
}

/// Waits for both readers to reach end of file, then drains what they sent
/// after the last poll and flushes the log, answering the captured stdout.
fn finish_output_capture(
    output_readers: [thread::JoinHandle<()>; 2],
    mut captured_output: CapturedOutput,
    stage: &str,
) -> Result<String, String> {
    for output_reader in output_readers {
        output_reader
            .join()
            .map_err(|_| format!("the {stage} output reader panicked"))?;
    }
    captured_output.drain()?;
    captured_output
        .command_log
        .flush()
        .map_err(|error| error.to_string())?;
    Ok(captured_output.captured_stdout)
}

fn create_command_log(log_path: &Path, command: &Command) -> Result<File, String> {
    let mut command_log = File::create(log_path)
        .map_err(|log_error| format!("cannot create command log {log_path:?}: {log_error}"))?;
    writeln!(command_log, "command={command:?}").map_err(|error| error.to_string())?;
    Ok(command_log)
}

/// A fresh process group lets the supervisor stop every descendant the
/// command starts, not only the direct child.
fn spawn_owned_process_group(command: &mut Command, stage: &str) -> Result<Child, String> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|launch_error| format!("could not start {stage}: {launch_error}"))
}

/// The command's output channel, its log and what it captured so far.
struct CapturedOutput {
    stream_receiver: Receiver<StreamEvent>,
    command_log: File,
    captured_stdout: String,
    output_style: OutputStyle,
}

impl CapturedOutput {
    fn drain(&mut self) -> Result<(), String> {
        drain_output(
            &self.stream_receiver,
            &mut self.command_log,
            &mut self.captured_stdout,
            self.output_style,
        )
    }
}

/// When the run started, when it last reported progress, and its budget.
struct SupervisionClock {
    started_at: Instant,
    last_progress_at: Instant,
    budget: Duration,
}

impl SupervisionClock {
    fn start(started_at: Instant, budget: Duration) -> Self {
        Self {
            started_at,
            last_progress_at: started_at,
            budget,
        }
    }

    fn is_budget_exhausted(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.started_at) >= self.budget
    }

    /// Answers whether a progress line is due at `now`, and when it is,
    /// restarts the progress interval from `now`.
    fn take_progress_due(&mut self, now: Instant) -> bool {
        let is_progress_due =
            now.saturating_duration_since(self.last_progress_at) >= PROGRESS_INTERVAL;
        if is_progress_due {
            self.last_progress_at = now;
        }
        is_progress_due
    }
}

/// The exit code to report: the command's own, 124 for a timeout (as
/// `timeout(1)` reports), or the signal's conventional status.
fn supervise_until_finished(
    child: &mut Child,
    stage: &str,
    signals: &SignalPipe,
    supervision_clock: &mut SupervisionClock,
    captured_output: &mut CapturedOutput,
) -> Result<i32, String> {
    loop {
        captured_output.drain()?;
        if let Some(received_signal) = signals.received_signal() {
            eprintln!(
                "[test-run] stage={stage} status=interrupted signal={}",
                process_signals::signal_name(received_signal)
            );
            return Ok(process_signals::signal_exit_status(received_signal));
        }
        match child.try_wait() {
            Ok(Some(exit_status)) => {
                return Ok(crate::disposable_target::exit_code_from_status(
                    &exit_status,
                ));
            }
            Ok(None) => {}
            Err(wait_error) => return Err(format!("cannot observe {stage}: {wait_error}")),
        }
        let now = Instant::now();
        if supervision_clock.is_budget_exhausted(now) {
            eprintln!(
                "[test-run] stage={stage} status=timeout budget_seconds={}",
                supervision_clock.budget.as_secs()
            );
            return Ok(124);
        }
        if supervision_clock.take_progress_due(now) {
            let elapsed = now.saturating_duration_since(supervision_clock.started_at);
            eprintln!(
                "[test-run] stage={stage} status=running progress_percent=0 elapsed_seconds={} eta_seconds=unknown budget_remaining_seconds={}",
                elapsed.as_secs(),
                supervision_clock.budget.saturating_sub(elapsed).as_secs()
            );
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn start_reader(
    stream: impl Read + Send + 'static,
    stdout: bool,
    stream_sender: Sender<StreamEvent>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for output_line in BufReader::new(stream).lines() {
            let stream_event = match output_line {
                Ok(text) => StreamEvent::Line { stdout, text },
                Err(read_error) => {
                    let _ = stream_sender.send(StreamEvent::Failed(format!(
                        "cannot read command output: {read_error}"
                    )));
                    return;
                }
            };
            if stream_sender.send(stream_event).is_err() {
                return;
            }
        }
    })
}

fn drain_output(
    stream_receiver: &Receiver<StreamEvent>,
    command_log: &mut File,
    captured_stdout: &mut String,
    output_style: OutputStyle,
) -> Result<(), String> {
    while let Ok(stream_event) = stream_receiver.try_recv() {
        match stream_event {
            StreamEvent::Failed(read_error) => return Err(read_error),
            StreamEvent::Line { stdout, text } => {
                writeln!(command_log, "{text}").map_err(|error| error.to_string())?;
                if stdout {
                    captured_stdout.push_str(&text);
                    captured_stdout.push('\n');
                }
                match console_echo_for_line(output_style, stdout, &text) {
                    ConsoleEcho::Stdout(echoed_text) => println!("{echoed_text}"),
                    ConsoleEcho::StderrVerbatim(echoed_text) => eprint!("{echoed_text}"),
                    ConsoleEcho::Stderr(echoed_text) => eprintln!("{echoed_text}"),
                    ConsoleEcho::Silent => {}
                }
            }
        }
    }
    Ok(())
}

/// Where one output line is echoed on the console.
#[derive(Debug, PartialEq, Eq)]
enum ConsoleEcho {
    Stdout(String),
    Stderr(String),
    /// Cargo's rendered diagnostics already end in a newline.
    StderrVerbatim(String),
    Silent,
}

/// Cargo's JSON messages are shown only as their rendered diagnostics, so
/// a build reads like a plain `cargo` run; artifact and build-script
/// messages have nothing rendered and stay silent. A stdout line that is
/// not JSON is an ordinary program line and passes through.
fn console_echo_for_line(
    output_style: OutputStyle,
    is_stdout: bool,
    line_text: &str,
) -> ConsoleEcho {
    match (output_style, is_stdout) {
        (OutputStyle::Quiet, _) => ConsoleEcho::Silent,
        (OutputStyle::Cargo, true) => match serde_json::from_str::<serde_json::Value>(line_text) {
            Ok(cargo_message) => match cargo_message["message"]["rendered"].as_str() {
                Some(rendered_diagnostic) => {
                    ConsoleEcho::StderrVerbatim(rendered_diagnostic.to_owned())
                }
                None => ConsoleEcho::Silent,
            },
            Err(_) => ConsoleEcho::Stdout(line_text.to_owned()),
        },
        (_, true) => ConsoleEcho::Stdout(line_text.to_owned()),
        (_, false) => ConsoleEcho::Stderr(line_text.to_owned()),
    }
}

fn stop_owned_process_group(process_group_id: i32, child: &mut Child) -> Result<(), String> {
    let _ = child.try_wait();
    if process_signals::process_group_is_alive(process_group_id) {
        process_signals::signal_process_group(process_group_id, libc::SIGTERM);
        let termination_started_at = Instant::now();
        while process_signals::process_group_is_alive(process_group_id)
            && termination_started_at.elapsed() < TERMINATION_GRACE
        {
            child.try_wait().map_err(|error| error.to_string())?;
            thread::sleep(POLL_INTERVAL);
        }
        if process_signals::process_group_is_alive(process_group_id) {
            process_signals::signal_process_group(process_group_id, libc::SIGKILL);
        }
    }
    child.wait().map_err(|error| error.to_string())?;
    if process_signals::process_group_is_alive(process_group_id) {
        return Err(format!(
            "owned process group {process_group_id} remains active; test evidence was retained"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "process_tests.rs"]
mod process_tests;
