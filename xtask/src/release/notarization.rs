//! `cargo xtask release notarize-dmg --dmg PATH --signing-identity NAME
//! --notary-profile PROFILE [--timeout-seconds N]`: signs, notarizes,
//! staples, and validates one immutable distribution image. Typed port of
//! the retired `notarize-dmg.sh`.
//!
//! The steps are planned as data first, so the exact Apple tool invocations
//! are tested without contacting the notary service. The timeout is a flag
//! rather than the shell's `NOTARIZATION_TIMEOUT_SECONDS` variable, so a
//! release run states it on the command line it logs.

use std::path::PathBuf;
use std::process::Command;

use super::system_step::{require_commands, run_system_step};

/// Apple notarization normally completes within about two minutes; three
/// bounds the wait without failing a slow but healthy submission.
pub(crate) const DEFAULT_NOTARIZATION_TIMEOUT_SECONDS: u64 = 180;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NotarizationRequest {
    pub(crate) disk_image: PathBuf,
    pub(crate) signing_identity: String,
    pub(crate) notary_profile: String,
    pub(crate) timeout_seconds: u64,
}

/// One planned step: its log name, the tool, and the tool's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedStep {
    pub(crate) step_name: &'static str,
    pub(crate) program: &'static str,
    pub(crate) program_arguments: Vec<String>,
}

pub(crate) fn notarization_plan(notarization_request: &NotarizationRequest) -> Vec<PlannedStep> {
    let disk_image = notarization_request.disk_image.display().to_string();
    vec![
        planned(
            "sign-dmg",
            "codesign",
            &[
                "--force",
                "--sign",
                &notarization_request.signing_identity,
                "--timestamp",
                &disk_image,
            ],
        ),
        planned(
            "submit-dmg-for-notarization",
            "xcrun",
            &[
                "notarytool",
                "submit",
                &disk_image,
                "--keychain-profile",
                &notarization_request.notary_profile,
                "--wait",
                "--timeout",
                &format!("{}s", notarization_request.timeout_seconds),
            ],
        ),
        planned(
            "staple-notarization-ticket",
            "xcrun",
            &["stapler", "staple", &disk_image],
        ),
        planned(
            "validate-stapled-ticket",
            "xcrun",
            &["stapler", "validate", &disk_image],
        ),
        planned(
            "assess-notarized-dmg",
            "spctl",
            &[
                "--assess",
                "--type",
                "open",
                "--context",
                "context:primary-signature",
                "--verbose=4",
                &disk_image,
            ],
        ),
    ]
}

fn planned(
    step_name: &'static str,
    program: &'static str,
    program_arguments: &[&str],
) -> PlannedStep {
    let mut owned_arguments = Vec::new();
    for program_argument in program_arguments {
        owned_arguments.push((*program_argument).to_owned());
    }
    PlannedStep {
        step_name,
        program,
        program_arguments: owned_arguments,
    }
}

pub(crate) fn validate_notarization_request(
    notarization_request: &NotarizationRequest,
) -> Result<(), String> {
    let has_contents = match std::fs::metadata(&notarization_request.disk_image) {
        Ok(image_metadata) => image_metadata.is_file() && image_metadata.len() > 0,
        Err(_) => false,
    };
    if !has_contents {
        return Err("--dmg must name a nonempty disk image".to_owned());
    }
    if notarization_request.timeout_seconds == 0 {
        return Err("--timeout-seconds must be positive".to_owned());
    }
    Ok(())
}

pub(crate) fn notarize_disk_image(
    notarization_request: &NotarizationRequest,
) -> Result<(), String> {
    validate_notarization_request(notarization_request)?;
    require_commands(&["codesign", "spctl", "xcrun"])?;
    for planned_step in notarization_plan(notarization_request) {
        run_system_step(
            planned_step.step_name,
            Command::new(planned_step.program).args(&planned_step.program_arguments),
        )?;
    }
    println!(
        "Notarized and stapled: {}",
        notarization_request.disk_image.display()
    );
    Ok(())
}

pub(crate) fn parse_timeout_seconds(timeout_text: Option<&String>) -> Result<u64, String> {
    let Some(timeout_text) = timeout_text else {
        return Ok(DEFAULT_NOTARIZATION_TIMEOUT_SECONDS);
    };
    match timeout_text.parse::<u64>() {
        Ok(timeout_seconds) => Ok(timeout_seconds),
        Err(_) => Err(format!(
            "--timeout-seconds must be a whole number of seconds (got {timeout_text:?})"
        )),
    }
}
