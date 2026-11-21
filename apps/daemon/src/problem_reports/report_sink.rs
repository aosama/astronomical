//! Where failures anywhere in the engine hand over a problem to report.
//!
//! The failures this covers surface deep inside the worker supervisor and
//! the download queue, far from the HTTP state that owns the report desk.
//! One sink registered once at startup, the way a tracing subscriber is,
//! keeps each failure site to a single line and threads no new parameter
//! through every worker and download constructor. Without a registered sink,
//! as in tests and tools, reporting a problem does nothing.

use std::sync::{Arc, OnceLock, RwLock};

use crate::ResolvedRuntimeConfig;

use super::{CatalogModelName, MacDescription, ProblemKind, ProblemReport, ProblemReportDesk};

static REGISTERED_REPORT_SINK: OnceLock<ReportSink> = OnceLock::new();

pub(crate) struct ReportSink {
    problem_report_desk: Arc<ProblemReportDesk>,
    live_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    catalog_huggingface_ids: Vec<String>,
    mac: MacDescription,
    runtime_handle: tokio::runtime::Handle,
}

impl ReportSink {
    /// The report for `problem`, naming the model only by its catalog id.
    pub(crate) fn build_report(
        &self,
        problem: ProblemKind,
        model_id: Option<&str>,
    ) -> ProblemReport {
        let mut model = None;
        if let Some(model_id) = model_id {
            model = Some(CatalogModelName::from_model_id(
                model_id,
                &self.catalog_huggingface_ids,
            ));
        }
        ProblemReport {
            app_version: env!("CARGO_PKG_VERSION").to_owned(),
            mac: self.mac.clone(),
            model,
            problem,
            note: None,
        }
    }

    fn problem_report_choice(&self) -> config::ProblemReportChoice {
        self.live_config
            .read()
            .map(|resolved_runtime_config| {
                resolved_runtime_config
                    .sharing_settings
                    .problem_report_choice
            })
            .unwrap_or_default()
    }
}

/// Registers the engine's one sink. It is skipped, and problems are not
/// reported, when this Mac cannot be described or no async runtime runs.
pub(crate) fn register_report_sink(
    problem_report_desk: Arc<ProblemReportDesk>,
    live_config: Arc<RwLock<ResolvedRuntimeConfig>>,
    download_catalog: &download_catalog::DownloadCatalog,
) {
    let (Some(mac), Ok(runtime_handle)) = (
        MacDescription::read_this_mac(),
        tokio::runtime::Handle::try_current(),
    ) else {
        tracing::warn!("problem reports are off: this Mac could not be described");
        return;
    };
    let mut catalog_huggingface_ids = Vec::new();
    for catalog_entry in download_catalog.entries() {
        catalog_huggingface_ids.push(catalog_entry.huggingface_id().to_owned());
    }
    let report_sink = ReportSink {
        problem_report_desk,
        live_config,
        catalog_huggingface_ids,
        mac,
        runtime_handle,
    };
    let _ = REGISTERED_REPORT_SINK.set(report_sink);
}

/// Hands a problem to the registered sink, which offers, sends or drops it
/// by the person's choice. Returns at once; sending runs in the background.
pub(crate) fn report_problem(problem: ProblemKind, model_id: Option<&str>) {
    let Some(report_sink) = REGISTERED_REPORT_SINK.get() else {
        return;
    };
    let problem_report = report_sink.build_report(problem, model_id);
    let problem_report_choice = report_sink.problem_report_choice();
    let problem_report_desk = Arc::clone(&report_sink.problem_report_desk);
    report_sink.runtime_handle.spawn(async move {
        problem_report_desk
            .offer(problem_report, problem_report_choice)
            .await;
    });
}

/// A failed model load, if it is worth a report.
pub(crate) fn report_model_load_failure(model_load_failure_reason: &str, requested_model_id: &str) {
    report_problem(
        ProblemKind::for_model_load_failure(model_load_failure_reason),
        Some(requested_model_id),
    );
}

/// A failed reply, if it is a fault rather than the request's own problem.
/// The loaded model is read from the worker's health snapshot.
pub(crate) fn report_reply_failure(
    failure_reason: &ipc_protocol::ChatGenerationFailureReason,
    health_snapshot: &RwLock<crate::WorkerHealthSnapshot>,
) {
    let Some(problem) = ProblemKind::for_reply_failure(failure_reason) else {
        return;
    };
    let ready_model_id = health_snapshot
        .read()
        .ok()
        .and_then(|worker_health_snapshot| worker_health_snapshot.ready_model_id.clone());
    report_problem(problem, ready_model_id.as_deref());
}

/// A failed catalog download, if it is a fault rather than an everyday cause.
pub(crate) fn report_download_failure(
    error_code: crate::library::DownloadJobPublicErrorCode,
    huggingface_id: &str,
) {
    if let Some(problem) = ProblemKind::for_download_failure(error_code) {
        report_problem(problem, Some(huggingface_id));
    }
}

/// The report for a problem the person wants to describe themselves, or
/// `None` when no sink is registered.
pub(crate) fn person_report(ready_model_id: Option<&str>) -> Option<ProblemReport> {
    let report_sink = REGISTERED_REPORT_SINK.get()?;
    Some(report_sink.build_report(ProblemKind::ReportedByPerson, ready_model_id))
}
