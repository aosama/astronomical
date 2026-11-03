//! Step lines in the shell pipeline's `<timestamp> step=NAME status=...`
//! shape, so logs and the CI searches written against them keep parsing.

use std::time::Instant;

use crate::disposable_target::local_timestamp;

pub(crate) struct StepTimer {
    step_name: &'static str,
    started_at: Instant,
}

pub(crate) fn start_step(step_name: &'static str) -> StepTimer {
    println!("{} step={step_name} status=start", local_timestamp());
    StepTimer {
        step_name,
        started_at: Instant::now(),
    }
}

impl StepTimer {
    pub(crate) fn finish(self, step_status: &str) {
        println!(
            "{} step={} status={step_status} elapsed_seconds={}",
            local_timestamp(),
            self.step_name,
            self.started_at.elapsed().as_secs()
        );
    }
}

pub(crate) fn log_line(log_text: &str) {
    println!("{} {log_text}", local_timestamp());
}
