use serde::Serialize;

use crate::{OpenAiResponse, OpenAiResponseOutputContent, OpenAiResponseOutputItem};

/// One semantic Server-Sent Events payload from the local Responses endpoint.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum OpenAiResponseStreamEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: OpenAiResponse,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: OpenAiResponse,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: OpenAiResponseOutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: OpenAiResponseOutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OpenAiResponseOutputContent,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OpenAiResponseOutputContent,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        text: String,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
        logprobs: Vec<serde_json::Value>,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
        logprobs: Vec<serde_json::Value>,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        name: String,
        arguments: String,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: OpenAiResponse,
    },
    #[serde(rename = "response.incomplete")]
    Incomplete {
        sequence_number: u64,
        response: OpenAiResponse,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: OpenAiResponse,
    },
    #[serde(rename = "error")]
    Error {
        sequence_number: u64,
        code: Option<String>,
        message: String,
        param: Option<String>,
    },
}

impl OpenAiResponseStreamEvent {
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::Created { .. } => "response.created",
            Self::InProgress { .. } => "response.in_progress",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::ReasoningSummaryTextDelta { .. } => "response.reasoning_summary_text.delta",
            Self::ReasoningSummaryTextDone { .. } => "response.reasoning_summary_text.done",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            Self::Completed { .. } => "response.completed",
            Self::Incomplete { .. } => "response.incomplete",
            Self::Failed { .. } => "response.failed",
            Self::Error { .. } => "error",
        }
    }

    #[must_use]
    pub const fn sequence_number(&self) -> u64 {
        match self {
            Self::Created {
                sequence_number, ..
            }
            | Self::InProgress {
                sequence_number, ..
            }
            | Self::OutputItemAdded {
                sequence_number, ..
            }
            | Self::OutputItemDone {
                sequence_number, ..
            }
            | Self::ContentPartAdded {
                sequence_number, ..
            }
            | Self::ContentPartDone {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDelta {
                sequence_number, ..
            }
            | Self::ReasoningSummaryTextDone {
                sequence_number, ..
            }
            | Self::OutputTextDelta {
                sequence_number, ..
            }
            | Self::OutputTextDone {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDelta {
                sequence_number, ..
            }
            | Self::FunctionCallArgumentsDone {
                sequence_number, ..
            }
            | Self::Completed {
                sequence_number, ..
            }
            | Self::Incomplete {
                sequence_number, ..
            }
            | Self::Failed {
                sequence_number, ..
            }
            | Self::Error {
                sequence_number, ..
            } => *sequence_number,
        }
    }
}
