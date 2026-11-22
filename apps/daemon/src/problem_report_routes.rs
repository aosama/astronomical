//! Wiring for problem reports and the daily count: which routes serve them,
//! behind which guard, and which instances get a report desk at all. Kept
//! apart from `application` so the core router stays readable.

use std::sync::{Arc, RwLock};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post, put},
};
use download_catalog::DownloadCatalog;

use crate::application::ApplicationState;
use crate::{ResolvedRuntimeConfig, ResolvedRuntimeConfigResolver};

/// The problem report endpoints (see `rest::problem_reports`), present only
/// when the instance has a report desk.
pub(crate) fn problem_report_routes(
    router: Router<ApplicationState>,
    has_problem_report_desk: bool,
) -> Router<ApplicationState> {
    if !has_problem_report_desk {
        return router;
    }
    let sharing_router = Router::new()
        .route(
            "/v1/problem-reports",
            get(crate::rest::problem_reports::list_problem_reports)
                .post(crate::rest::problem_reports::start_person_report),
        )
        .route(
            "/v1/problem-reports/{report_number}/send",
            post(crate::rest::problem_reports::send_problem_report)
                .layer(DefaultBodyLimit::max(8 * 1024)),
        )
        .route(
            "/v1/problem-reports/{report_number}/decline",
            post(crate::rest::problem_reports::decline_problem_report),
        )
        .route(
            "/v1/daily-count",
            get(crate::rest::daily_count::show_daily_count),
        )
        .route(
            "/v1/config/sharing",
            put(crate::rest::sharing_setting::update_sharing_setting)
                .layer(DefaultBodyLimit::max(4 * 1024)),
        )
        // Sending a report and changing what is shared answer only the chat
        // window, this instance's Observatory page and tools that send no
        // Origin. The chat window's CORS rule also admits any page served
        // from a loopback port, which must not switch sharing on or send a
        // waiting report.
        .route_layer(axum::middleware::from_fn(
            crate::rest::loopback_browser_guard::refuse_foreign_browser_requests_except_thin_talk,
        ));
    router.merge(sharing_router)
}

/// The production report desk: also registered as the engine's one report
/// sink, so failures in the worker and the download queue reach it.
pub(crate) fn registered_problem_report_desk(
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    reloadable_config: &Arc<RwLock<ResolvedRuntimeConfig>>,
    download_catalog: &DownloadCatalog,
) -> Option<Arc<crate::problem_reports::ProblemReportDesk>> {
    let problem_report_desk = problem_report_desk_for(runtime_config_resolver)?;
    crate::problem_reports::register_report_sink(
        Arc::clone(&problem_report_desk),
        Arc::clone(reloadable_config),
        download_catalog,
    );
    Some(problem_report_desk)
}

/// The report desk for an instance, logging sent reports in its state
/// folder and sending to the receiver compiled into this build.
///
/// The Mac App Store build gets none: its privacy manifest and its App
/// Privacy answers in App Store Connect declare that no data is collected,
/// so until both are updated (issue #2413, "Mac App Store version") that
/// build must offer no report, register no sink and serve no report route.
pub(crate) fn problem_report_desk_for(
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
) -> Option<Arc<crate::problem_reports::ProblemReportDesk>> {
    if runtime_config_resolver
        .instance_paths()
        .keeps_state_in_app_container()
    {
        return None;
    }
    Some(Arc::new(crate::problem_reports::ProblemReportDesk::new(
        runtime_config_resolver.state_directory(),
        crate::problem_reports::ReportReceiver::compiled_in(),
    )))
}
