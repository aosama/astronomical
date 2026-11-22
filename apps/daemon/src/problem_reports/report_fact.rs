//! One extra fact on a report: a fixed name with a value that is a number
//! or one of a fixed set of words, never text taken from an error or a log.

use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ReportFact {
    /// The name the developers group by, such as `engine_exit`.
    pub(crate) key: &'static str,
    /// The words the person reads before the value, such as "Engine exit".
    #[serde(skip)]
    pub(crate) label: &'static str,
    pub(crate) value: String,
}

impl ReportFact {
    pub(crate) fn new(key: &'static str, label: &'static str, value: impl Into<String>) -> Self {
        Self {
            key,
            label,
            value: value.into(),
        }
    }

    pub(crate) fn readable_line(&self) -> String {
        format!("{}: {}", self.label, self.value)
    }
}
