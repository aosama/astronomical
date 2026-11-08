//! Reveals one daemon-chosen path in Finder.
//!
//! Callers resolve the path themselves from daemon-owned state (the active
//! configuration file, a catalog model's own directory); a page never supplies
//! a path, so a crafted request cannot point Finder anywhere else.

use std::{path::Path, process::Stdio, time::Duration};

use axum::http::StatusCode;
use tokio::{process::Command, time::timeout};

/// `open -R` returns as soon as Finder accepts the request; the bound only
/// stops a hung Launch Services call from holding an HTTP request open.
const FINDER_REVEAL_TIMEOUT: Duration = Duration::from_secs(5);

/// Selects the path in a Finder window: 204 when Finder accepted it, 500 when
/// `open` failed, timed out, or could not start.
pub(crate) async fn reveal_path_in_finder(revealed_path: &Path) -> StatusCode {
    let reveal_outcome = timeout(
        FINDER_REVEAL_TIMEOUT,
        Command::new("/usr/bin/open")
            .arg("-R")
            .arg(revealed_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
    )
    .await;
    match reveal_outcome {
        Ok(Ok(exit_status)) if exit_status.success() => StatusCode::NO_CONTENT,
        Ok(Ok(_)) | Ok(Err(_)) | Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}
