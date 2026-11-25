//! Progress for `cargo xtask measure-speed`: a line every five seconds with
//! the step, the percentage, the elapsed time and the time left, as every
//! long-running job in this repository reports.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

struct ProgressState {
    total_steps: usize,
    completed_steps: AtomicUsize,
    stage_text: Mutex<String>,
    is_finished: AtomicBool,
    started_at: Instant,
}

pub struct SpeedProgress {
    progress_state: Arc<ProgressState>,
    reporter_thread: Option<JoinHandle<()>>,
}

impl SpeedProgress {
    pub fn start(total_steps: usize) -> Self {
        let progress_state = Arc::new(ProgressState {
            total_steps: total_steps.max(1),
            completed_steps: AtomicUsize::new(0),
            stage_text: Mutex::new("starting".to_owned()),
            is_finished: AtomicBool::new(false),
            started_at: Instant::now(),
        });
        let reporter_state = Arc::clone(&progress_state);
        let reporter_thread = std::thread::spawn(move || report_until_finished(&reporter_state));
        Self {
            progress_state,
            reporter_thread: Some(reporter_thread),
        }
    }

    pub fn set_stage(&self, stage_text: &str) {
        if let Ok(mut current_stage) = self.progress_state.stage_text.lock() {
            stage_text.clone_into(&mut current_stage);
        }
    }

    pub fn complete_step(&self) {
        self.skip_steps(1);
    }

    /// Counts the steps of a case that cannot run (its model did not load) as
    /// done, so the time left keeps describing the cases still to come.
    pub fn skip_steps(&self, step_count: usize) {
        self.progress_state
            .completed_steps
            .fetch_add(step_count, Ordering::Relaxed);
    }

    pub fn finish(mut self) {
        self.progress_state
            .is_finished
            .store(true, Ordering::Relaxed);
        if let Some(reporter_thread) = self.reporter_thread.take() {
            reporter_thread.join().ok();
        }
        println!(
            "[measure-speed] status=finished elapsed_seconds={}",
            self.progress_state.started_at.elapsed().as_secs()
        );
    }
}

fn report_until_finished(progress_state: &ProgressState) {
    let mut last_report_at = Instant::now();
    while !progress_state.is_finished.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
        if last_report_at.elapsed() >= PROGRESS_INTERVAL {
            last_report_at = Instant::now();
            println!("{}", progress_line(progress_state));
        }
    }
}

fn progress_line(progress_state: &ProgressState) -> String {
    let completed_steps = progress_state.completed_steps.load(Ordering::Relaxed);
    let elapsed_seconds = progress_state.started_at.elapsed().as_secs();
    let percent_done = completed_steps * 100 / progress_state.total_steps;
    let eta_text = (completed_steps > 0).then(|| {
        let remaining_steps = progress_state.total_steps.saturating_sub(completed_steps) as u64;
        format!(
            "{}",
            elapsed_seconds * remaining_steps / completed_steps as u64
        )
    });
    let stage_text = progress_state
        .stage_text
        .lock()
        .map(|stage| stage.clone())
        .unwrap_or_default();
    format!(
        "[measure-speed] progress={percent_done}% step={completed_steps}/{} elapsed_seconds={elapsed_seconds} eta_seconds={} stage={stage_text}",
        progress_state.total_steps,
        eta_text.unwrap_or_else(|| "unknown".to_owned()),
    )
}
