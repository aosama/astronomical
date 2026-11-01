//! Embedded single-page admin console served by the daemon.
//!
//! The Observatory console is embedded directly into the `astronomicald` binary
//! through `include_str!` so the daemon has no filesystem dependency and no
//! separate frontend build step. The console shell and its assets live at
//! `apps/daemon/console/` and are reached via `../../console/...`; the console
//! renders no model output — conversations live in Thin Talk, reached through
//! the `astronomical://chat` link the console's chat view offers.

use axum::{Router, body::Body, http::header, response::Response, routing::get};

const INDEX_HTML: &str = include_str!("../../console/index.html");
const CONSOLE_JS: &str = include_str!("../../console/console.js");
const LIBRARY_JS: &str = include_str!("../../console/library.js");
const LIBRARY_RENDER_JS: &str = include_str!("../../console/library-render.js");
const OVERVIEW_COMPACT_JS: &str = include_str!("../../console/overview-compact.js");
const MEMORY_CONTROL_JS: &str = include_str!("../../console/memory-control.js");
const CONNECT_JS: &str = include_str!("../../console/connect.js");
const EXPERTS_JS: &str = include_str!("../../console/experts.js");
const EXPERTS_RENDER_JS: &str = include_str!("../../console/experts-render.js");
const CONSOLE_CSS: &str = include_str!("../../console/console.css");
const LIBRARY_CSS: &str = include_str!("../../console/library.css");
const EXPERTS_CSS: &str = include_str!("../../console/experts.css");

const HTML_CONTENT_TYPE: &str = "text/html; charset=utf-8";
const JAVASCRIPT_CONTENT_TYPE: &str = "application/javascript; charset=utf-8";
const CSS_CONTENT_TYPE: &str = "text/css; charset=utf-8";

/// Returns the Observatory console routes, ready to merge into any daemon
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
        .route("/experts", get(console_index))
        .route("/library", get(console_index))
        .route("/connect", get(console_index))
        .route("/settings", get(console_index))
        .route("/console.js", get(console_script))
        .route("/library.js", get(library_script))
        .route("/library-render.js", get(library_render_script))
        .route("/overview-compact.js", get(overview_compact_script))
        .route("/memory-control.js", get(memory_control_script))
        .route("/connect.js", get(connect_script))
        .route("/experts.js", get(experts_script))
        .route("/experts-render.js", get(experts_render_script))
        .route("/console.css", get(console_stylesheet))
        .route("/library.css", get(library_stylesheet))
        .route("/experts.css", get(experts_stylesheet))
}

/// `GET /` — the Observatory single-page shell. References the per-view scripts and
/// `/console.css`, each served from this module, so a browser loads behavior and
/// styling with no build step.
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

pub(crate) async fn library_render_script() -> Response {
    embedded_text_response(LIBRARY_RENDER_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn overview_compact_script() -> Response {
    embedded_text_response(OVERVIEW_COMPACT_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn memory_control_script() -> Response {
    embedded_text_response(MEMORY_CONTROL_JS, JAVASCRIPT_CONTENT_TYPE)
}

/// `GET /connect.js` — the connection material that teaches a user how to point an
/// OpenAI-compatible coding agent at this Mac. It derives every published value from
/// the console origin, so a copied sample matches the port the running instance
/// actually listens on rather than a port baked in when the sample was written.
pub(crate) async fn connect_script() -> Response {
    embedded_text_response(CONNECT_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn experts_script() -> Response {
    embedded_text_response(EXPERTS_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn experts_render_script() -> Response {
    embedded_text_response(EXPERTS_RENDER_JS, JAVASCRIPT_CONTENT_TYPE)
}

/// `GET /experts.css` — the Experts view styling (library bar, layer rows,
/// session totals, and the measurement disclosure).
pub(crate) async fn experts_stylesheet() -> Response {
    embedded_text_response(EXPERTS_CSS, CSS_CONTENT_TYPE)
}

/// `GET /console.css` — the Observatory dark-mode styling with large fonts.
pub(crate) async fn console_stylesheet() -> Response {
    embedded_text_response(CONSOLE_CSS, CSS_CONTENT_TYPE)
}

pub(crate) async fn library_stylesheet() -> Response {
    embedded_text_response(LIBRARY_CSS, CSS_CONTENT_TYPE)
}

fn embedded_text_response(body: &'static str, content_type: &'static str) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    // Embedded assets change with every released binary, so a cached copy always
    // shows stale UI; forbid caching outright rather than managing revalidation.
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An inline SVG carrying only a viewBox collapses to 0x0 inside a flex
    /// button, which shipped navigation icons as empty circles. Every SVG the
    /// console styles must therefore carry an explicit CSS size.
    #[test]
    fn console_css_sizes_every_styled_svg() {
        assert!(
            CONSOLE_CSS.contains(".navigation-button svg"),
            "console.css is missing a sizing rule for `.navigation-button svg`"
        );
        let navigation_icon_rule = CONSOLE_CSS
            .split(".navigation-button svg")
            .nth(1)
            .expect("navigation icon rule must exist");
        assert!(
            navigation_icon_rule.contains("width:") && navigation_icon_rule.contains("height:"),
            "navigation icon rule must set explicit width and height"
        );
    }

    /// The console renders no model output anymore, so the shell must not load
    /// any render-stack asset or playground script; chat is handed off to Thin
    /// Talk through the astronomical:// link with a menu-bar fallback.
    #[test]
    fn console_hands_chat_to_thin_talk_without_a_render_stack() {
        for stale_reference in ["/render/", "playground"] {
            assert!(
                !INDEX_HTML.contains(stale_reference),
                "console shell still references the retired `{stale_reference}` surface"
            );
        }
        for thin_talk_handoff in ["astronomical://chat", "Open Thin Talk", "choose Chat"] {
            assert!(
                INDEX_HTML.contains(thin_talk_handoff),
                "console shell is missing the Thin Talk handoff `{thin_talk_handoff}`"
            );
        }
    }

    #[tokio::test]
    async fn render_assets_are_no_longer_served() {
        use tower::ServiceExt;

        let app = console_routes::<()>();
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/render/canvas-render.js")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn embedded_responses_forbid_caching() {
        let response = console_stylesheet().await;
        let cache_control = response
            .headers()
            .get(header::CACHE_CONTROL)
            .expect("cache-control header must be present");
        assert_eq!(cache_control, "no-store");
    }
}
