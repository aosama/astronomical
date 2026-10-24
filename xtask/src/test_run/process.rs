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

#[derive(Clone, Copy)]
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

pub(super) fn run_command(
    command: &mut Command,
    log_path: &Path,
    stage: &str,
    budget: Duration,
    output_style: OutputStyle,
    signals: &SignalPipe,
) -> Result<CommandOutput, String> {
    let mut command_log = File::create(log_path)
        .map_err(|log_error| format!("cannot create command log {log_path:?}: {log_error}"))?;
    writeln!(command_log, "command={command:?}").map_err(|error| error.to_string())?;
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|launch_error| format!("could not start {stage}: {launch_error}"))?;
    let process_group_id = i32::try_from(child.id())
        .map_err(|_| "the child process identifier exceeds the supported range".to_owned())?;
    let (stream_sender, stream_receiver) = mpsc::channel();
    let stdout_reader = start_reader(
        child.stdout.take().expect("piped stdout"),
        true,
        stream_sender.clone(),
    );
    let stderr_reader = start_reader(
        child.stderr.take().expect("piped stderr"),
        false,
        stream_sender,
    );
    let started_at = Instant::now();
    let mut last_progress_at = started_at;
    let mut captured_stdout = String::new();
    eprintln!(
        "[test-run] stage={stage} status=start progress_percent=0 eta_seconds=unknown budget_seconds={}",
        budget.as_secs()
    );
    let command_outcome = loop {
        if let Err(read_error) = drain_output(
            &stream_receiver,
            &mut command_log,
            &mut captured_stdout,
            output_style,
        ) {
            break Err(read_error);
        }
        if let Some(received_signal) = signals.received_signal() {
            eprintln!(
                "[test-run] stage={stage} status=interrupted signal={}",
                process_signals::signal_name(received_signal)
            );
            break Ok(process_signals::signal_exit_status(received_signal));
        }
        match child.try_wait() {
            Ok(Some(exit_status)) => {
                break Ok(crate::disposable_target::exit_code_from_status(
                    &exit_status,
                ));
            }
            Ok(None) => {}
            Err(wait_error) => break Err(format!("cannot observe {stage}: {wait_error}")),
        }
        if started_at.elapsed() >= budget {
            eprintln!(
                "[test-run] stage={stage} status=timeout budget_seconds={}",
                budget.as_secs()
            );
            break Ok(124);
        }
        if last_progress_at.elapsed() >= PROGRESS_INTERVAL {
            eprintln!(
                "[test-run] stage={stage} status=running progress_percent=0 elapsed_seconds={} eta_seconds=unknown budget_remaining_seconds={}",
                started_at.elapsed().as_secs(),
                budget.saturating_sub(started_at.elapsed()).as_secs()
            );
            last_progress_at = Instant::now();
        }
        thread::sleep(POLL_INTERVAL);
    };
    stop_owned_process_group(process_group_id, &mut child)?;
    for output_reader in [stdout_reader, stderr_reader] {
        output_reader
            .join()
            .map_err(|_| format!("the {stage} output reader panicked"))?;
    }
    drain_output(
        &stream_receiver,
        &mut command_log,
        &mut captured_stdout,
        output_style,
    )?;
    command_log.flush().map_err(|error| error.to_string())?;
    let exit_code = command_outcome?;
    eprintln!(
        "[test-run] stage={stage} status=finished progress_percent=100 elapsed_seconds={} eta_seconds=0 exit_code={exit_code}",
        started_at.elapsed().as_secs()
    );
    Ok(CommandOutput {
        exit_code,
        stdout: captured_stdout,
    })
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
                match (output_style, stdout) {
                    (OutputStyle::Quiet, _) => {}
                    (OutputStyle::Cargo, true) => {
                        if let Ok(cargo_message) = serde_json::from_str::<serde_json::Value>(&text)
                        {
                            if let Some(diagnostic) = cargo_message["message"]["rendered"].as_str()
                            {
                                eprint!("{diagnostic}");
                            }
                        } else {
                            println!("{text}");
                        }
                    }
                    (_, true) => println!("{text}"),
                    (_, false) => eprintln!("{text}"),
                }
            }
        }
    }
    Ok(())
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
