//! The engine's last few events, kept so a report can say what led up to a
//! problem. Events are a fixed set carrying numbers only: no model names,
//! paths or text, so a report can show them as they are.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

/// How many events are kept: enough to cover a load and a few replies before
/// a failure, few enough to read on the report card.
pub(crate) const MAXIMUM_KEPT_EVENTS: usize = 12;

/// Bytes in one GB as Astronomical counts it (2^30), matching the screens.
const BYTES_PER_GB: f64 = 1_073_741_824.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FlightEvent {
    ModelLoadStarted,
    ModelLoaded,
    ModelLoadFailed,
    ReplyFinished {
        prompt_token_count: u32,
        generated_token_count: u16,
    },
    ReplyFailed,
    MemoryLimitChanged {
        memory_limit_bytes: u64,
    },
    EngineExited,
}

impl FlightEvent {
    fn plain_description(self) -> String {
        match self {
            Self::ModelLoadStarted => "model load started".to_owned(),
            Self::ModelLoaded => "model loaded".to_owned(),
            Self::ModelLoadFailed => "model load failed".to_owned(),
            Self::ReplyFinished {
                prompt_token_count,
                generated_token_count,
            } => format!(
                "reply finished ({prompt_token_count} tokens in, {generated_token_count} out)"
            ),
            Self::ReplyFailed => "reply failed".to_owned(),
            Self::MemoryLimitChanged { memory_limit_bytes } => format!(
                "memory limit set to {:.1} GB",
                memory_limit_bytes as f64 / BYTES_PER_GB
            ),
            Self::EngineExited => "engine exited".to_owned(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct FlightRecorder {
    kept_events: Mutex<VecDeque<(Instant, FlightEvent)>>,
}

impl FlightRecorder {
    pub(crate) fn record_at(&self, happened_at: Instant, flight_event: FlightEvent) {
        let mut kept_events = self
            .kept_events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        kept_events.push_back((happened_at, flight_event));
        if kept_events.len() > MAXIMUM_KEPT_EVENTS {
            kept_events.pop_front();
        }
    }

    /// The kept events, oldest first, each as "12 s before: model loaded".
    pub(crate) fn lines_before(&self, report_time: Instant) -> Vec<String> {
        let kept_events = self
            .kept_events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut event_lines = Vec::with_capacity(kept_events.len());
        for (happened_at, flight_event) in kept_events.iter() {
            let seconds_before = report_time
                .saturating_duration_since(*happened_at)
                .as_secs();
            event_lines.push(format!(
                "{seconds_before} s before: {}",
                flight_event.plain_description()
            ));
        }
        event_lines
    }
}

/// The engine's one recorder. A process-wide recorder, like the report sink,
/// lets each event site record with one line.
static ENGINE_FLIGHT_RECORDER: FlightRecorder = FlightRecorder {
    kept_events: Mutex::new(VecDeque::new()),
};

pub(crate) fn record_flight_event(flight_event: FlightEvent) {
    ENGINE_FLIGHT_RECORDER.record_at(Instant::now(), flight_event);
}

/// Records the start of a model load. Progress updates republish a load in
/// flight many times, so only its first appearance counts as the start.
pub(crate) fn record_model_load_start(was_load_in_flight: bool, is_load_in_flight: bool) {
    if !was_load_in_flight && is_load_in_flight {
        record_flight_event(FlightEvent::ModelLoadStarted);
    }
}

pub(crate) fn recent_flight_event_lines() -> Vec<String> {
    ENGINE_FLIGHT_RECORDER.lines_before(Instant::now())
}

#[cfg(test)]
#[path = "flight_recorder_tests.rs"]
mod flight_recorder_tests;
