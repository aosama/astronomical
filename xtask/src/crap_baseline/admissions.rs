//! `crap-admissions.json`: the tracked list of functions allowed above the
//! new-function complexity cap.
//!
//! Admissions live in a reviewed file rather than on the command line so
//! each exception is a one-line diff with its reason beside it, and so the
//! pre-commit check, which judges against the committed baseline, accepts
//! the same exceptions the regeneration did.

use std::collections::BTreeSet;

use serde_json::Value;

pub(crate) const ADMISSIONS_RELATIVE_PATH: &str = "crap-admissions.json";

/// Parses the admissions document into `file:function` labels. Every entry
/// needs a non-empty reason, because an exception nobody explained is the
/// kind a later reader removes or copies blindly.
pub(crate) fn parse_admissions(admissions_text: &str) -> Result<BTreeSet<String>, String> {
    let admissions_document: Value = serde_json::from_str(admissions_text).map_err(|error| {
        format!("[crap-baseline] status=failed reason=malformed-admissions error={error}")
    })?;
    let admitted_entries = admissions_document
        .get("admitted_functions")
        .and_then(Value::as_array)
        .ok_or("[crap-baseline] status=failed reason=admissions-without-admitted-functions")?;
    let mut admitted_function_labels = BTreeSet::new();
    for admitted_entry in admitted_entries {
        let admitted_label = admitted_entry
            .get("function")
            .and_then(Value::as_str)
            .filter(|label_text| label_text.contains(".rs:"))
            .ok_or("[crap-baseline] status=failed reason=admission-needs-file-and-function")?;
        let has_reason = admitted_entry
            .get("reason")
            .and_then(Value::as_str)
            .is_some_and(|reason_text| !reason_text.trim().is_empty());
        if !has_reason {
            return Err(format!(
                "[crap-baseline] status=failed reason=admission-without-reason function={admitted_label}"
            ));
        }
        if !admitted_function_labels.insert(admitted_label.to_owned()) {
            return Err(format!(
                "[crap-baseline] status=failed reason=duplicate-admission function={admitted_label}"
            ));
        }
    }
    Ok(admitted_function_labels)
}
