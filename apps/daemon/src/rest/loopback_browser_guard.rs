//! Refuses browser requests that did not come from this instance's own page.
//!
//! Binding to loopback keeps other machines out, but a web page on this Mac can
//! still reach the daemon: DNS rebinding makes a hostile domain resolve here, which
//! the browser then treats as same-origin, and the CORS policy admits any
//! `http://localhost` port. That matters for every endpoint that acts on this
//! Mac (reading local paths, rewriting `config.json`, starting downloads,
//! revealing folders in Finder, moving models to the Trash), so they check both
//! headers. Tools with no browser send no `Origin` and name a loopback `Host`,
//! so they pass.

use axum::{
    Json,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};

const LOOPBACK_HOST_NAMES: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

fn is_port_number(port_text: &str) -> bool {
    if port_text.is_empty() {
        return false;
    }
    for port_byte in port_text.bytes() {
        if !port_byte.is_ascii_digit() {
            return false;
        }
    }
    true
}

fn names_loopback(host: &str) -> bool {
    let (host_name, port_text) = match host.rsplit_once(':') {
        // An IPv6 literal's own colons sit inside its brackets.
        Some((host_name, port_text)) if !port_text.ends_with(']') => (host_name, Some(port_text)),
        _ => (host, None),
    };
    let is_valid_port = match port_text {
        Some(port_text) => is_port_number(port_text),
        None => true,
    };
    is_valid_port && LOOPBACK_HOST_NAMES.contains(&host_name)
}

/// `Origin` must be exactly this instance's own page, not merely some loopback
/// origin, because another local web server is a different site.
fn request_comes_from_this_instance(host: Option<&str>, origin: Option<&str>) -> bool {
    match (host, origin) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(host), None) => names_loopback(host),
        (Some(host), Some(origin)) => {
            names_loopback(host) && origin.strip_prefix("http://") == Some(host)
        }
    }
}

/// `None` when the header is present but not text, which no browser would send.
fn header_text(headers: &HeaderMap, header_name: header::HeaderName) -> Option<Option<&str>> {
    match headers.get(header_name) {
        None => Some(None),
        Some(header_value) => match header_value.to_str() {
            Ok(header_text) => Some(Some(header_text)),
            Err(_) => None,
        },
    }
}

/// The Thin Talk canvas's page origin. It is a custom URL scheme that only the
/// Mac app's own web view serves, and browsers set `Origin` themselves, so no
/// web page can claim it; the download routes accept it because Astro reads
/// and starts downloads from that canvas.
pub(crate) const THIN_TALK_CANVAS_ORIGIN: &str = "thintalk-asset://shell";

/// The refusal to answer with, or `None` when the request may proceed.
pub(crate) fn refusal_for_foreign_request(headers: &HeaderMap) -> Option<Response> {
    refusal_unless_from(headers, &[])
}

/// As `refusal_for_foreign_request`, but also admitting the named app origins,
/// still only with a loopback `Host`, so a rebound domain stays refused.
fn refusal_unless_from(headers: &HeaderMap, admitted_app_origins: &[&str]) -> Option<Response> {
    let is_allowed = match (
        header_text(headers, header::HOST),
        header_text(headers, header::ORIGIN),
    ) {
        (Some(host), Some(origin)) => {
            request_comes_from_this_instance(host, origin)
                || is_admitted_app_request(host, origin, admitted_app_origins)
        }
        _ => false,
    };
    if is_allowed {
        return None;
    }
    Some(
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "status": "forbidden_origin",
                "message": "this endpoint only answers this instance's own Observatory page",
            })),
        )
            .into_response(),
    )
}

fn is_admitted_app_request(
    host: Option<&str>,
    origin: Option<&str>,
    admitted_app_origins: &[&str],
) -> bool {
    match (host, origin) {
        (Some(host), Some(origin)) => {
            names_loopback(host) && admitted_app_origins.contains(&origin)
        }
        _ => false,
    }
}

/// The same refusal as a route layer, for routers whose every endpoint acts on
/// this Mac (Finder reveals, moving models to the Trash), so a new handler on
/// such a router cannot forget the check.
pub(crate) async fn refuse_foreign_browser_requests(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    match refusal_for_foreign_request(request.headers()) {
        Some(refusal) => refusal,
        None => next.run(request).await,
    }
}

/// The route layer for the download controls: this instance's own page and
/// the Thin Talk canvas, whose Astro reads status and starts downloads.
pub(crate) async fn refuse_foreign_browser_requests_except_thin_talk(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    match refusal_unless_from(request.headers(), &[THIN_TALK_CANVAS_ORIGIN]) {
        Some(refusal) => refusal,
        None => next.run(request).await,
    }
}

#[cfg(test)]
#[path = "loopback_browser_guard_tests.rs"]
mod tests;
