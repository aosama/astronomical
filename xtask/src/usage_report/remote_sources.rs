//! The network edges of the usage report: the `gh` CLI for GitHub, which
//! reuses the developer's existing sign-in, and `curl` for the public Hugging
//! Face API, which needs none. Everything they return is parsed elsewhere.

use std::process::Command;

/// Runs `gh` with `gh_arguments` and returns its standard output.
pub fn gh_output(gh_arguments: &[&str]) -> Result<String, String> {
    let gh_answer = Command::new("gh")
        .args(gh_arguments)
        .output()
        .map_err(|start_error| format!("gh failed to start: {start_error}"))?;
    if !gh_answer.status.success() {
        return Err(format!(
            "gh {} failed: {}",
            gh_arguments.first().copied().unwrap_or_default(),
            String::from_utf8_lossy(&gh_answer.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&gh_answer.stdout).into_owned())
}

/// The Hugging Face models API response for one repository, with the 30-day
/// and all-time download counts and likes expanded.
pub fn hugging_face_model_response(repository_id: &str) -> Result<String, String> {
    let model_url = format!(
        "https://huggingface.co/api/models/{repository_id}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes"
    );
    // --globoff keeps curl from reading the `[]` in the query as a URL range.
    let curl_answer = Command::new("curl")
        .args([
            "--silent",
            "--fail",
            "--globoff",
            "--max-time",
            "20",
            &model_url,
        ])
        .output()
        .map_err(|start_error| format!("curl failed to start: {start_error}"))?;
    if !curl_answer.status.success() {
        return Err(format!("Hugging Face did not answer for {repository_id}"));
    }
    Ok(String::from_utf8_lossy(&curl_answer.stdout).into_owned())
}
