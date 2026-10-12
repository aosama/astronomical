//! Embedded chat-answer render stack served by the Observatory console.
//!
//! This is the markdown / KaTeX / Mermaid / code renderer plus the model-output
//! sanitiser that the chat transcript renders with. Every file is embedded with
//! `include_str!` / `include_bytes!` so the supervisor keeps no filesystem
//! dependency and needs no frontend build step.
//!
//! The files are symlinks into Thin Talk's canvas shell
//! (`apps/thin-talk/Sources/ThinTalkCanvas/Resources/web/`), which is the single
//! source of truth. The modules are bridge-agnostic: they depend only on the
//! `window.__thintalk*` namespace, the vendor libraries, and the DOM, so the same
//! files serve both the Thin Talk native canvas and the Observatory console. Editing
//! the symlinks here would be a drift bug — change the files under Thin Talk instead.

use axum::{
    Router,
    body::Body,
    extract::Path,
    http::{StatusCode, header, response::Response},
    routing::get,
};

/// Text render assets, keyed by the path served under `/render/`.
const CANVAS_TRUST_JS: &str = include_str!("../console/render/canvas-trust.js");
const CANVAS_CONTENT_KEY_JS: &str = include_str!("../console/render/canvas-content-key.js");
const CANVAS_MATH_JS: &str = include_str!("../console/render/canvas-math.js");
const CANVAS_DIAGRAMS_JS: &str = include_str!("../console/render/canvas-diagrams.js");
const CANVAS_RENDER_JS: &str = include_str!("../console/render/canvas-render.js");

const MORPHDOM_JS: &str = include_str!("../console/render/vendor/morphdom-umd.min.js");
const PURIFY_JS: &str = include_str!("../console/render/vendor/purify.min.js");
const MARKED_JS: &str = include_str!("../console/render/vendor/marked.min.js");
const MERMAID_JS: &str = include_str!("../console/render/vendor/mermaid.min.js");
const HIGHLIGHT_JS: &str = include_str!("../console/render/vendor/highlight.min.js");
const HIGHLIGHT_CSS: &str = include_str!("../console/render/vendor/highlight-github-dark.min.css");
const KATEX_JS: &str = include_str!("../console/render/vendor/katex/katex.min.js");
const KATEX_CSS: &str = include_str!("../console/render/vendor/katex/katex.min.css");

/// KaTeX webfonts. Embedded as raw bytes because `.woff2` is not valid UTF-8.
const FONT_AMS_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_AMS-Regular.woff2");
const FONT_CALIGRAPHIC_BOLD: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Caligraphic-Bold.woff2");
const FONT_CALIGRAPHIC_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Caligraphic-Regular.woff2");
const FONT_FRAKTUR_BOLD: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Fraktur-Bold.woff2");
const FONT_FRAKTUR_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Fraktur-Regular.woff2");
const FONT_MAIN_BOLD: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Main-Bold.woff2");
const FONT_MAIN_BOLD_ITALIC: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Main-BoldItalic.woff2");
const FONT_MAIN_ITALIC: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Main-Italic.woff2");
const FONT_MAIN_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Main-Regular.woff2");
const FONT_MATH_BOLD_ITALIC: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Math-BoldItalic.woff2");
const FONT_MATH_ITALIC: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Math-Italic.woff2");
const FONT_SANS_SERIF_BOLD: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_SansSerif-Bold.woff2");
const FONT_SANS_SERIF_ITALIC: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_SansSerif-Italic.woff2");
const FONT_SANS_SERIF_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_SansSerif-Regular.woff2");
const FONT_SCRIPT_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Script-Regular.woff2");
const FONT_SIZE1_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Size1-Regular.woff2");
const FONT_SIZE2_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Size2-Regular.woff2");
const FONT_SIZE3_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Size3-Regular.woff2");
const FONT_SIZE4_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Size4-Regular.woff2");
const FONT_TYPEWRITER_REGULAR: &[u8] =
    include_bytes!("../console/render/vendor/katex/fonts/KaTeX_Typewriter-Regular.woff2");

const JAVASCRIPT_CONTENT_TYPE: &str = "application/javascript; charset=utf-8";
const CSS_CONTENT_TYPE: &str = "text/css; charset=utf-8";
const WOFF2_CONTENT_TYPE: &str = "font/woff2";

/// The `/render/` routes, merged into the console router.
pub(crate) fn render_routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route("/render/{*path}", get(render_asset))
}

/// `GET /render/{*path}` — one handler for the embedded render stack. Text assets
/// (JS/CSS) and webfonts (`.woff2`) share this route; everything else is a 404.
/// The catch-all captures the path *after* `/render/`, so matchers key on the
/// relative asset path (e.g. `vendor/katex/katex.min.js`), not the full URL.
pub(crate) async fn render_asset(Path(asset_path): Path<String>) -> Response<Body> {
    if let Some((body, content_type)) = text_asset(&asset_path) {
        return embedded_text_response(body, content_type);
    }
    if let Some(stem) = asset_path
        .strip_prefix("vendor/katex/fonts/")
        .and_then(|name| name.strip_suffix(".woff2"))
    {
        if let Some(bytes) = font_asset(stem) {
            return embedded_bytes_response(bytes, WOFF2_CONTENT_TYPE);
        }
    }
    not_found()
}

fn text_asset(relative: &str) -> Option<(&'static str, &'static str)> {
    let body = match relative {
        "canvas-trust.js" => CANVAS_TRUST_JS,
        "canvas-content-key.js" => CANVAS_CONTENT_KEY_JS,
        "canvas-math.js" => CANVAS_MATH_JS,
        "canvas-diagrams.js" => CANVAS_DIAGRAMS_JS,
        "canvas-render.js" => CANVAS_RENDER_JS,
        "vendor/morphdom-umd.min.js" => MORPHDOM_JS,
        "vendor/purify.min.js" => PURIFY_JS,
        "vendor/marked.min.js" => MARKED_JS,
        "vendor/mermaid.min.js" => MERMAID_JS,
        "vendor/highlight.min.js" => HIGHLIGHT_JS,
        "vendor/highlight-github-dark.min.css" => HIGHLIGHT_CSS,
        "vendor/katex/katex.min.js" => KATEX_JS,
        "vendor/katex/katex.min.css" => KATEX_CSS,
        _ => return None,
    };
    let content_type = if relative.ends_with(".css") {
        CSS_CONTENT_TYPE
    } else {
        JAVASCRIPT_CONTENT_TYPE
    };
    Some((body, content_type))
}

fn font_asset(stem: &str) -> Option<&'static [u8]> {
    let bytes: &'static [u8] = match stem {
        "KaTeX_AMS-Regular" => FONT_AMS_REGULAR,
        "KaTeX_Caligraphic-Bold" => FONT_CALIGRAPHIC_BOLD,
        "KaTeX_Caligraphic-Regular" => FONT_CALIGRAPHIC_REGULAR,
        "KaTeX_Fraktur-Bold" => FONT_FRAKTUR_BOLD,
        "KaTeX_Fraktur-Regular" => FONT_FRAKTUR_REGULAR,
        "KaTeX_Main-Bold" => FONT_MAIN_BOLD,
        "KaTeX_Main-BoldItalic" => FONT_MAIN_BOLD_ITALIC,
        "KaTeX_Main-Italic" => FONT_MAIN_ITALIC,
        "KaTeX_Main-Regular" => FONT_MAIN_REGULAR,
        "KaTeX_Math-BoldItalic" => FONT_MATH_BOLD_ITALIC,
        "KaTeX_Math-Italic" => FONT_MATH_ITALIC,
        "KaTeX_SansSerif-Bold" => FONT_SANS_SERIF_BOLD,
        "KaTeX_SansSerif-Italic" => FONT_SANS_SERIF_ITALIC,
        "KaTeX_SansSerif-Regular" => FONT_SANS_SERIF_REGULAR,
        "KaTeX_Script-Regular" => FONT_SCRIPT_REGULAR,
        "KaTeX_Size1-Regular" => FONT_SIZE1_REGULAR,
        "KaTeX_Size2-Regular" => FONT_SIZE2_REGULAR,
        "KaTeX_Size3-Regular" => FONT_SIZE3_REGULAR,
        "KaTeX_Size4-Regular" => FONT_SIZE4_REGULAR,
        "KaTeX_Typewriter-Regular" => FONT_TYPEWRITER_REGULAR,
        _ => return None,
    };
    Some(bytes)
}

fn embedded_text_response(body: &'static str, content_type: &'static str) -> Response<Body> {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    mark_as_uncacheable(&mut response);
    response
}

fn embedded_bytes_response(body: &'static [u8], content_type: &'static str) -> Response<Body> {
    let mut response = Response::new(Body::from(body.to_vec()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    mark_as_uncacheable(&mut response);
    response
}

fn mark_as_uncacheable(response: &mut Response<Body>) {
    // Embedded assets change with every released binary, so a cached copy always
    // shows stale UI; forbid caching outright rather than managing revalidation.
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
}

fn not_found() -> Response<Body> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Body::empty())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use tower::ServiceExt;

    /// Issues a single GET against the render router and returns the response.
    /// The router is stateless, so the unit state type is enough to drive it.
    async fn get_render_asset(path: &str) -> axum::response::Response {
        let app = super::render_routes::<()>();
        app.oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn serves_canvas_modules_as_javascript() {
        let response = get_render_asset("/render/canvas-trust.js").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/javascript; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn serves_vendor_stylesheets_as_css() {
        let response = get_render_asset("/render/vendor/katex/katex.min.css").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/css; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn serves_katex_webfonts_as_woff2() {
        let response =
            get_render_asset("/render/vendor/katex/fonts/KaTeX_Main-Regular.woff2").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "font/woff2"
        );
    }

    #[tokio::test]
    async fn serves_mermaid_as_javascript() {
        let response = get_render_asset("/render/vendor/mermaid.min.js").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/javascript; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn returns_not_found_for_unknown_render_paths() {
        let response = get_render_asset("/render/does-not-exist.js").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
