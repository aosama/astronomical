//! Embedded single-page admin console served by the daemon.
//!
//! The Observatory console is embedded directly into the `astronomicald` binary
//! through `include_str!` so the daemon has no filesystem dependency and no
//! separate frontend build step. The console shell and its assets live at
//! `apps/daemon/console/` and are reached via `../../console/...`; the console
//! renders no model output — conversations live in Thin Talk, reached through
//! the channel's own chat link the console's chat view offers.

use axum::{Router, body::Body, http::header, response::Response, routing::get};

const INDEX_HTML: &str = include_str!("../../console/index.html");
const CONSOLE_JS: &str = include_str!("../../console/console.js");
const APPEARANCE_JS: &str = include_str!("../../console/appearance.js");
const DOWNLOAD_MODELS_JS: &str = include_str!("../../console/download-models.js");
const DOWNLOAD_MODELS_RENDER_JS: &str = include_str!("../../console/download-models-render.js");
const DOWNLOAD_MODELS_GROUPS_JS: &str = include_str!("../../console/download-models-groups.js");
const DOWNLOAD_MODELS_ROW_JS: &str = include_str!("../../console/download-models-row.js");
const DOWNLOAD_MODELS_LINKED_JS: &str = include_str!("../../console/download-models-linked.js");
const DOWNLOAD_MODELS_INSTALLED_JS: &str =
    include_str!("../../console/download-models-installed.js");
const DOWNLOAD_MODELS_REMOVAL_JS: &str = include_str!("../../console/download-models-removal.js");
const OVERVIEW_JS: &str = include_str!("../../console/overview.js");
const OVERVIEW_CSS: &str = include_str!("../../console/overview.css");
const NEURAL_ENGINE_PANEL_JS: &str = include_str!("../../console/neural-engine-panel.js");
const MEMORY_CONTROL_JS: &str = include_str!("../../console/memory-control.js");
const CONNECT_JS: &str = include_str!("../../console/connect.js");
const EXPERTS_JS: &str = include_str!("../../console/experts.js");
const EXPERTS_RENDER_JS: &str = include_str!("../../console/experts-render.js");
const CONFIGURATION_MODEL_JS: &str = include_str!("../../console/configuration-model.js");
const CONFIGURATION_LAYOUT_JS: &str = include_str!("../../console/configuration-layout.js");
const CONFIGURATION_JS: &str = include_str!("../../console/configuration.js");
const CONFIGURATION_CONTROLS_JS: &str = include_str!("../../console/configuration-controls.js");
const CONFIGURATION_PAGES_JS: &str = include_str!("../../console/configuration-pages.js");
const CONFIGURATION_CSS: &str = include_str!("../../console/configuration.css");
const CONSOLE_CSS: &str = include_str!("../../console/console.css");
const DOWNLOAD_MODELS_CSS: &str = include_str!("../../console/download-models.css");
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
        // The Model page was folded into Overview; its old address still serves
        // the shell, and console.js opens Overview from it.
        .route("/model", get(console_index))
        .route("/experts", get(console_index))
        .route("/download-models", get(console_index))
        // The page was renamed Download Models; its old address still serves
        // the shell, and console.js opens the renamed page from it.
        .route("/library", get(console_index))
        .route("/connect", get(console_index))
        .route("/settings", get(console_index))
        .route("/tuning", get(console_index))
        // The Configuration page was split into Settings and Tuning; its old address
        // still serves the shell, and console.js opens Settings from it.
        .route("/configuration", get(console_index))
        .route("/configuration-model.js", get(configuration_model_script))
        .route("/configuration-layout.js", get(configuration_layout_script))
        .route("/configuration.js", get(configuration_script))
        .route(
            "/configuration-controls.js",
            get(configuration_controls_script),
        )
        .route("/configuration-pages.js", get(configuration_pages_script))
        .route("/configuration.css", get(configuration_stylesheet))
        .route("/console.js", get(console_script))
        .route("/appearance.js", get(appearance_script))
        .route("/download-models.js", get(download_models_script))
        .route(
            "/download-models-render.js",
            get(download_models_render_script),
        )
        .route(
            "/download-models-groups.js",
            get(download_models_groups_script),
        )
        .route("/download-models-row.js", get(download_models_row_script))
        .route(
            "/download-models-linked.js",
            get(download_models_linked_script),
        )
        .route(
            "/download-models-installed.js",
            get(download_models_installed_script),
        )
        .route(
            "/download-models-removal.js",
            get(download_models_removal_script),
        )
        .route("/overview.js", get(overview_script))
        .route("/overview.css", get(overview_stylesheet))
        .route("/neural-engine-panel.js", get(neural_engine_panel_script))
        .route("/memory-control.js", get(memory_control_script))
        .route("/connect.js", get(connect_script))
        .route("/experts.js", get(experts_script))
        .route("/experts-render.js", get(experts_render_script))
        .route("/console.css", get(console_stylesheet))
        .route("/download-models.css", get(download_models_stylesheet))
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

/// `GET /appearance.js` — the light, dark or follow-the-Mac palette choice.
pub(crate) async fn appearance_script() -> Response {
    embedded_text_response(APPEARANCE_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_render_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_RENDER_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_groups_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_GROUPS_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_row_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_ROW_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_linked_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_LINKED_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_installed_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_INSTALLED_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn download_models_removal_script() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_REMOVAL_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn overview_script() -> Response {
    embedded_text_response(OVERVIEW_JS, JAVASCRIPT_CONTENT_TYPE)
}

/// `GET /overview.css` — the Overview view's layout and colours.
pub(crate) async fn overview_stylesheet() -> Response {
    embedded_text_response(OVERVIEW_CSS, CSS_CONTENT_TYPE)
}

pub(crate) async fn neural_engine_panel_script() -> Response {
    embedded_text_response(NEURAL_ENGINE_PANEL_JS, JAVASCRIPT_CONTENT_TYPE)
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

pub(crate) async fn configuration_model_script() -> Response {
    embedded_text_response(CONFIGURATION_MODEL_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn configuration_script() -> Response {
    embedded_text_response(CONFIGURATION_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn configuration_layout_script() -> Response {
    embedded_text_response(CONFIGURATION_LAYOUT_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn configuration_controls_script() -> Response {
    embedded_text_response(CONFIGURATION_CONTROLS_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn configuration_pages_script() -> Response {
    embedded_text_response(CONFIGURATION_PAGES_JS, JAVASCRIPT_CONTENT_TYPE)
}

pub(crate) async fn configuration_stylesheet() -> Response {
    embedded_text_response(CONFIGURATION_CSS, CSS_CONTENT_TYPE)
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

pub(crate) async fn download_models_stylesheet() -> Response {
    embedded_text_response(DOWNLOAD_MODELS_CSS, CSS_CONTENT_TYPE)
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
    /// Talk through the channel's own link with a menu-bar fallback. The shell
    /// carries no scheme, so it cannot send one channel to another's window.
    #[test]
    fn console_hands_chat_to_thin_talk_without_a_render_stack() {
        for stale_reference in ["/render/", "playground"] {
            assert!(
                !INDEX_HTML.contains(stale_reference),
                "console shell still references the retired `{stale_reference}` surface"
            );
        }
        assert!(
            !INDEX_HTML.contains("astronomical://"),
            "console shell must not hardcode a channel's chat link"
        );
        for thin_talk_handoff in ["open-thin-talk", "Open the chat window", "choose Chat"] {
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
