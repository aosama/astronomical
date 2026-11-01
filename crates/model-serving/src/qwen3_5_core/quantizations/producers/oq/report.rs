use serde_json::Value;
use thiserror::Error;

/// File name oMLX oQ/oQe writes at the artifact root.
pub(crate) const OQ_IMATRIX_REPORT_FILE_NAME: &str = "oq_imatrix_report.json";

// Measured artifacts run ~35 KB for ~500 modules because the report lists
// every applied module by name; 1 MiB leaves room for far larger models while
// still refusing a pathological file before it is parsed.
const MAXIMUM_OQ_REPORT_BYTES: usize = 1024 * 1024;

/// Provenance recovered from an oQ/oQe imatrix report. It records how the
/// producer calibrated; it never constrains how the artifact executes, so
/// every field is optional and a sparse or newer report still reads.
///
/// The report's cache path is deliberately not retained: it names the
/// producer's machine, which has no meaning on the serving host.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OqImatrixProvenance {
    pub calibration_dataset: Option<String>,
    pub is_imatrix_enabled: Option<bool>,
    pub entry_count: Option<u64>,
    pub applied_module_count: Option<u64>,
    pub missing_module_count: Option<u64>,
    pub mismatched_module_count: Option<u64>,
}

/// A report that cannot be read as provenance.
#[derive(Debug, Error)]
pub enum OqImatrixReportError {
    #[error("oQ imatrix report is {actual_size_bytes} bytes, exceeding {maximum_size_bytes}")]
    ReportTooLarge {
        actual_size_bytes: usize,
        maximum_size_bytes: usize,
    },
    #[error("failed to decode oQ imatrix report JSON")]
    DeserializeReport(#[source] serde_json::Error),
    #[error("oQ imatrix report is not a JSON object")]
    ReportIsNotAnObject,
}

impl OqImatrixProvenance {
    /// Parses the bounded report leniently: unknown keys are ignored and a
    /// key of an unexpected type reads as absent rather than failing, because
    /// the producer owns this format and may extend it at any release.
    pub fn from_json_bytes(report_bytes: &[u8]) -> Result<Self, OqImatrixReportError> {
        if report_bytes.len() > MAXIMUM_OQ_REPORT_BYTES {
            return Err(OqImatrixReportError::ReportTooLarge {
                actual_size_bytes: report_bytes.len(),
                maximum_size_bytes: MAXIMUM_OQ_REPORT_BYTES,
            });
        }
        let report_value: Value = serde_json::from_slice(report_bytes)
            .map_err(OqImatrixReportError::DeserializeReport)?;
        let Some(report_object) = report_value.as_object() else {
            return Err(OqImatrixReportError::ReportIsNotAnObject);
        };
        Ok(Self {
            calibration_dataset: report_object
                .get("calib_dataset")
                .and_then(Value::as_str)
                .map(str::to_owned),
            is_imatrix_enabled: report_object.get("enabled").and_then(Value::as_bool),
            entry_count: report_object.get("entry_count").and_then(Value::as_u64),
            applied_module_count: report_object
                .get("applied")
                .and_then(count_of_listed_modules),
            missing_module_count: report_object
                .get("missing")
                .and_then(count_of_listed_modules),
            mismatched_module_count: report_object
                .get("mismatched")
                .and_then(count_of_listed_modules),
        })
    }
}

/// Reads a module tally that the producer writes either as the module-name
/// list itself or as a bare count.
fn count_of_listed_modules(tally: &Value) -> Option<u64> {
    match tally {
        Value::Array(module_names) => u64::try_from(module_names.len()).ok(),
        Value::Number(count) => count.as_u64(),
        _ => None,
    }
}

#[cfg(test)]
mod report_tests;
