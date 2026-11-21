//! Problem reports: the short, fixed-shape reports a person may choose to
//! send the developers when something goes wrong (issue #2413).
//!
//! A report is built only from fixed fields: the app version, the Mac's chip,
//! memory and macOS version, a catalog model id, the kind of problem and two
//! memory figures. It never carries free text from an error, a log line, a
//! prompt or a path, so there is nothing personal for a scrubber to miss. A
//! model that is not in the download catalog is named only as such, because
//! a folder or model someone made themselves can carry a personal name.
//!
//! These modules build and describe reports and never touch the network;
//! sending lives apart so the person's choice is checked in one place.

mod catalog_model_name;
mod mac_description;
mod problem_kind;
mod problem_report;
mod problem_report_desk;
mod report_receiver;
mod sent_report_log;
mod waiting_reports;

pub(crate) use catalog_model_name::CatalogModelName;
pub(crate) use mac_description::MacDescription;
pub(crate) use problem_kind::ProblemKind;
pub(crate) use problem_report::{MemoryShortfall, ProblemReport};
pub(crate) use problem_report_desk::{ProblemReportDesk, ReportSendOutcome};
pub(crate) use report_receiver::{ReportReceiver, sentry_envelope_body};
pub(crate) use sent_report_log::{SentProblemReport, SentReportLog};
pub(crate) use waiting_reports::{WaitingProblemReport, WaitingReports};
