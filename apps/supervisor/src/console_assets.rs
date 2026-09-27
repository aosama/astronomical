//! Embedded single-page admin console served by the supervisor.
//!
//! The Observatory console is embedded directly into the `astronomicald` binary
//! through `include_str!` so the supervisor has no filesystem dependency and no
//! separate frontend build step. The console shell and its assets live at
//! `apps/supervisor/console/` and are reached via `../console/...`; the chat
//! answer render stack is served by `console_render_assets` (merged in below).

use axum::{Router, body::Body, http::header, response::Response, routing::get};

const INDEX_HTML: &str = include_str!("../console/index.html");
const CONSOLE_JS: &str = include_str!("../console/console.js");
const LIBRARY_JS: &str = include_str!("../console/library.js");
const LIBRARY_RENDER_JS: &str = include_str!("../console/library-render.js");
const OVERVIEW_COMPACT_JS: &str = include_str!("../console/overview-compact.js");
const MEMORY_CONTROL_JS: &str = include_str!("../console/memory-control.js");
const CONNECT_JS: &str = include_str!("../console/connect.js");
const PLAYGROUND_JS: &str = include_str!("../console/playground.js");
const CONSOLE_CSS: &str = include_str!("../console/console.css");
const LIBRARY_CSS: &str = include_str!("../console/library.css");

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
        .route("/connect", get(console_index))
        .route("/settings", get(console_index))
        .route("/console.js", get(console_script))
        .route("/library.js", get(library_script))
        .route("/library-render.js", get(library_render_script))
        .route("/overview-compact.js", get(overview_compact_script))
        .route("/memory-control.js", get(memory_control_script))
        .route("/connect.js", get(connect_script))
        .route("/playground.js", get(playground_script))
        .route("/console.css", get(console_stylesheet))
        .route("/library.css", get(library_stylesheet))
        .merge(crate::console_render_assets::render_routes())
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

pub(crate) async fn playground_script() -> Response {
    embedded_text_response(PLAYGROUND_JS, JAVASCRIPT_CONTENT_TYPE)
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
    /// button, which shipped composer icons as empty circles. Every SVG the
    /// console styles must therefore carry an explicit CSS size.
    #[test]
    fn console_css_sizes_every_styled_svg() {
        for css_rule_selector in [
            ".navigation-button svg",
            ".effort-pill__chevron",
            ".composer-round-button svg,",
            ".composer-send-button svg",
        ] {
            assert!(
                CONSOLE_CSS.contains(css_rule_selector),
                "console.css is missing a sizing rule for `{css_rule_selector}`"
            );
        }
        let composer_icon_rule = CONSOLE_CSS
            .split(".composer-round-button svg,")
            .nth(1)
            .expect("composer icon rule must exist");
        assert!(
            composer_icon_rule.contains("width:") && composer_icon_rule.contains("height:"),
            "composer icon rule must set explicit width and height"
        );
    }

    /// The text-size control sets an inline font-size on #chat-transcript; if a
    /// chat-content rule pins its own fixed size, the control visibly does
    /// nothing. The transcript declares the default, and message content
    /// (messages, markdown, role/meta lines) must inherit or use em units.
    #[test]
    fn chat_text_scales_with_the_text_size_control() {
        assert!(
            CONSOLE_CSS.contains(".chat-transcript:empty::before")
                && CONSOLE_CSS
                    .rsplit_once(".chat-transcript:empty::before")
                    .is_some_and(|(_, after)| after.contains("font-size: inherit;")),
            "the empty-transcript placeholder must inherit the controlled size"
        );
        for rule in CONSOLE_CSS
            .split('}')
            .filter_map(|chunk| chunk.split_once('{'))
        {
            let (selectors, body) = rule;
            let is_chat_content = selectors.contains(".chat-message")
                || selectors.contains(".markdown-body")
                || selectors.contains(".message__");
            if !is_chat_content {
                continue;
            }
            for declaration in body.lines().filter(|line| line.contains("font-size:")) {
                assert!(
                    declaration.contains("inherit") || declaration.trim_end().ends_with("em;"),
                    "chat-content rule pins a fixed size, breaking the text-size control: \
                     selectors `{selectors}` declaration `{declaration}`"
                );
            }
        }
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
