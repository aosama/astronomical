//! Embedded single-page admin console served by the supervisor.
//!
//! The Observatory console is embedded directly in the supervisor binary.
//! included directly into the `astronomicald` binary through `include_str!`
//! so the supervisor has no filesystem dependency and no separate frontend
//! build step. The assets live at `apps/supervisor/console/` and are reached
//! via `../console/...` from this source file.

use axum::{Router, body::Body, http::header, response::Response, routing::get};

const INDEX_HTML: &str = include_str!("../console/index.html");
const CONSOLE_JS: &str = include_str!("../console/console.js");
const LIBRARY_JS: &str = include_str!("../console/library.js");
const OVERVIEW_COMPACT_JS: &str = include_str!("../console/overview-compact.js");
const MEMORY_CONTROL_JS: &str = include_str!("../console/memory-control.js");
const PLAYGROUND_JS: &str = include_str!("../console/playground.js");
const CONSOLE_CSS: &str = include_str!("../console/console.css");

const HTML_CONTENT_TYPE: &str = "text/html; charset=utf-8";
const JAVASCRIPT_CONTENT_TYPE: &str = "application/javascript; charset=utf-8";
const CSS_CONTENT_TYPE: &str = "text/css; charset=utf-8";

/// Returns the Observatory console routes, ready to merge into any supervisor
/// `Router` without repeating asset routes across every builder variant.
pub(crate) fn console_routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/", get(console_index))
        .route("/overview", get(console_index))
        .route("/chat", get(console_index))
        .route("/model", get(console_index))
        .route("/library", get(console_index))
        .route("/settings", get(console_index))
        .route("/console.js", get(console_script))
        .route("/library.js", get(library_script))
        .route("/overview-compact.js", get(overview_compact_script))
        .route("/memory-control.js", get(memory_control_script))
        .route("/playground.js", get(playground_script))
        .route("/console.css", get(console_stylesheet))
}

/// `GET /` — the Observatory single-page shell. References `/console.js` and
/// `/console.css` so a browser loads behavior and styling with no build step.
pub(crate) async fn console_index() -> Response {
    embedded_text_response(INDEX_HTML, HTML_CONTENT_TYPE)
}

/// `GET /console.js` — the Observatory behavior (vanilla JavaScript, no
/// framework, no build step).
pub(crate) async fn console_script() -> Response {
    embedded_text_response(CONSOLE_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn library_script() -> Response {
    embedded_text_response(LIBRARY_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn overview_compact_script() -> Response {
    embedded_text_response(OVERVIEW_COMPACT_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn memory_control_script() -> Response {
    embedded_text_response(MEMORY_CONTROL_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn playground_script() -> Response {
    embedded_text_response(PLAYGROUND_JS, JAVASCRIPT_CONTENT_TYPE)
}

/// `GET /console.css` — the Observatory dark-mode styling with large fonts.
pub(crate) async fn console_stylesheet() -> Response {
    embedded_text_response(CONSOLE_CSS, CSS_CONTENT_TYPE)
}

fn embedded_text_response(body: &'static str, content_type: &'static str) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    response
}
