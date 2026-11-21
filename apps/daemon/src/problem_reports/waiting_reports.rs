//! Reports offered to the person and waiting for "Send report" or "Don't
//! send". They live only in memory: a report the person never answered is
//! dropped when the engine restarts rather than sent later by surprise.

use std::sync::Mutex;

use serde::Serialize;

use super::ProblemReport;

/// At most this many reports wait at once, the newest kept. Someone shown a
/// stack of reports reads none of them, and a repeating failure would
/// otherwise grow the stack without bound.
pub(crate) const MAXIMUM_WAITING_REPORTS: usize = 3;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WaitingProblemReport {
    pub(crate) report_number: u64,
    pub(crate) report: ProblemReport,
}

#[derive(Debug, Default)]
pub(crate) struct WaitingReports {
    waiting_state: Mutex<WaitingState>,
}

#[derive(Debug, Default)]
struct WaitingState {
    reports: Vec<WaitingProblemReport>,
    last_report_number: u64,
}

impl WaitingReports {
    /// Adds `report` unless an identical one is already waiting, and returns
    /// the number it waits under. The same failure repeating is one report.
    pub(crate) fn offer(&self, report: ProblemReport) -> u64 {
        let mut waiting_state = lock_waiting_state(&self.waiting_state);
        for waiting_report in &waiting_state.reports {
            if waiting_report.report == report {
                return waiting_report.report_number;
            }
        }
        waiting_state.last_report_number += 1;
        let report_number = waiting_state.last_report_number;
        waiting_state.reports.push(WaitingProblemReport {
            report_number,
            report,
        });
        if waiting_state.reports.len() > MAXIMUM_WAITING_REPORTS {
            waiting_state.reports.remove(0);
        }
        report_number
    }

    /// Returns a report that was taken but did not go, under its own number,
    /// so the person's screen still points at it when they try again.
    pub(crate) fn put_back(&self, report_number: u64, report: ProblemReport) {
        let mut waiting_state = lock_waiting_state(&self.waiting_state);
        let insert_position = waiting_state
            .reports
            .partition_point(|waiting_report| waiting_report.report_number < report_number);
        waiting_state
            .reports
            .insert(insert_position, WaitingProblemReport { report_number, report });
        if waiting_state.reports.len() > MAXIMUM_WAITING_REPORTS {
            waiting_state.reports.remove(0);
        }
    }

    pub(crate) fn waiting(&self) -> Vec<WaitingProblemReport> {
        lock_waiting_state(&self.waiting_state).reports.clone()
    }

    /// Removes and returns the report waiting under `report_number`; both
    /// sending and declining a report take it, so it is answered once.
    pub(crate) fn take(&self, report_number: u64) -> Option<ProblemReport> {
        let mut waiting_state = lock_waiting_state(&self.waiting_state);
        let report_position = waiting_state
            .reports
            .iter()
            .position(|waiting_report| waiting_report.report_number == report_number)?;
        Some(waiting_state.reports.remove(report_position).report)
    }
}

/// A poisoned lock only means another thread panicked mid-update of this
/// small list; the list itself is still valid, so keep serving it.
fn lock_waiting_state(
    waiting_state: &Mutex<WaitingState>,
) -> std::sync::MutexGuard<'_, WaitingState> {
    waiting_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
#[path = "waiting_reports_tests.rs"]
mod waiting_reports_tests;
