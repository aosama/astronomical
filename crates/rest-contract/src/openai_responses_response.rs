use std::collections::BTreeMap;

use serde::Serialize;

/// One complete response returned by the local Responses endpoint.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    completed_at: Option<u64>,
    status: OpenAiResponseStatus,
    error: Option<OpenAiResponseError>,
    incomplete_details: Option<OpenAiResponseIncompleteDetails>,
    instructions: Option<String>,
    metadata: BTreeMap<String, String>,
    model: String,
    output: Vec<OpenAiResponseOutputItem>,
    output_text: String,
    parallel_tool_calls: bool,
    temperature: Option<f32>,
    tool_choice: &'static str,
    tools: Vec<OpenAiResponseFunctionTool>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    previous_response_id: Option<String>,
    truncation: &'static str,
    usage: Option<OpenAiResponseUsage>,
}

/// Validated request settings echoed by every lifecycle snapshot of a response.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiResponseRequestConfiguration {
    pub metadata: BTreeMap<String, String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub tool_choice: &'static str,
    pub tools: Vec<OpenAiResponseFunctionTool>,
}

impl Default for OpenAiResponseRequestConfiguration {
    fn default() -> Self {
        Self {
            metadata: BTreeMap::new(),
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            tool_choice: "auto",
            tools: Vec::new(),
        }
    }
}

impl OpenAiResponse {
    /// Applies the validated request settings represented by this response.
    #[must_use]
    pub fn with_request_configuration(
        mut self,
        request_configuration: &OpenAiResponseRequestConfiguration,
    ) -> Self {
        self.metadata.clone_from(&request_configuration.metadata);
        self.temperature = request_configuration.temperature;
        self.top_p = request_configuration.top_p;
        self.max_output_tokens = request_configuration.max_output_tokens;
        self.tool_choice = request_configuration.tool_choice;
        self.tools.clone_from(&request_configuration.tools);
        self
    }

    /// Creates the empty snapshot carried by initial streaming lifecycle events.
    #[must_use]
    pub fn in_progress(
        response_id: impl Into<String>,
        created_at: u64,
        model_id: impl Into<String>,
        instructions: Option<String>,
    ) -> Self {
        Self {
            id: response_id.into(),
            object: "response",
            created_at,
            completed_at: None,
            status: OpenAiResponseStatus::InProgress,
            error: None,
            incomplete_details: None,
            instructions,
            metadata: BTreeMap::new(),
            model: model_id.into(),
            output: Vec::new(),
            output_text: String::new(),
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: "auto",
            tools: Vec::new(),
            top_p: None,
            max_output_tokens: None,
            previous_response_id: None,
            truncation: "disabled",
            usage: None,
        }
    }

    /// Creates one completed local response with default request-echo fields.
    #[must_use]
    pub fn completed(
        response_id: impl Into<String>,
        created_at: u64,
        completed_at: u64,
        model_id: impl Into<String>,
        instructions: Option<String>,
        output: Vec<OpenAiResponseOutputItem>,
        usage: OpenAiResponseUsage,
    ) -> Self {
        let output_text = output
            .iter()
            .filter_map(OpenAiResponseOutputItem::message_text)
            .collect::<String>();
        Self {
            id: response_id.into(),
            object: "response",
            created_at,
            completed_at: Some(completed_at),
            status: OpenAiResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions,
            metadata: BTreeMap::new(),
            model: model_id.into(),
            output,
            output_text,
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: "auto",
            tools: Vec::new(),
            top_p: None,
            max_output_tokens: None,
            previous_response_id: None,
            truncation: "disabled",
            usage: Some(usage),
        }
    }

    /// Creates one response interrupted by its generated-token ceiling.
    #[must_use]
    pub fn incomplete_at_output_token_limit(
        response_id: impl Into<String>,
        created_at: u64,
        model_id: impl Into<String>,
        instructions: Option<String>,
        mut output: Vec<OpenAiResponseOutputItem>,
        usage: OpenAiResponseUsage,
    ) -> Self {
        for output_item in &mut output {
            output_item.mark_incomplete();
        }
        let mut response = Self::completed(
            response_id,
            created_at,
            created_at,
            model_id,
            instructions,
            output,
            usage,
        );
        response.completed_at = None;
        response.status = OpenAiResponseStatus::Incomplete;
        response.incomplete_details = Some(OpenAiResponseIncompleteDetails {
            reason: "max_output_tokens",
        });
        response
    }

    /// Creates one failed response after the local worker has reported a request failure.
    #[must_use]
    pub fn failed(
        response_id: impl Into<String>,
        created_at: u64,
        model_id: impl Into<String>,
        instructions: Option<String>,
        mut output: Vec<OpenAiResponseOutputItem>,
        error_code: impl Into<String>,
        error_message: impl Into<String>,
    ) -> Self {
        for output_item in &mut output {
            output_item.mark_incomplete();
        }
        let output_text = output
            .iter()
            .filter_map(OpenAiResponseOutputItem::message_text)
            .collect::<String>();
        Self {
            id: response_id.into(),
            object: "response",
            created_at,
            completed_at: None,
            status: OpenAiResponseStatus::Failed,
            error: Some(OpenAiResponseError {
                code: error_code.into(),
                message: error_message.into(),
            }),
            incomplete_details: None,
            instructions,
            metadata: BTreeMap::new(),
            model: model_id.into(),
            output,
            output_text,
            parallel_tool_calls: true,
            temperature: None,
            tool_choice: "auto",
            tools: Vec::new(),
            top_p: None,
            max_output_tokens: None,
            previous_response_id: None,
            truncation: "disabled",
            usage: None,
        }
    }
}

/// The lifecycle state of a Responses object.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Cancelled,
}

/// One model-produced output item.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiResponseOutputItem {
    Reasoning {
        id: String,
        summary: Vec<OpenAiResponseReasoningSummary>,
        content: Vec<OpenAiResponseReasoningContent>,
        status: OpenAiResponseItemStatus,
    },
    Message {
        id: String,
        role: &'static str,
        content: Vec<OpenAiResponseOutputContent>,
        status: OpenAiResponseItemStatus,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: OpenAiResponseItemStatus,
    },
}

impl OpenAiResponseOutputItem {
    #[must_use]
    pub fn reasoning_in_progress(id: impl Into<String>) -> Self {
        Self::Reasoning {
            id: id.into(),
            summary: Vec::new(),
            content: Vec::new(),
            status: OpenAiResponseItemStatus::InProgress,
        }
    }

    #[must_use]
    pub fn message_in_progress(id: impl Into<String>) -> Self {
        Self::Message {
            id: id.into(),
            role: "assistant",
            content: Vec::new(),
            status: OpenAiResponseItemStatus::InProgress,
        }
    }

    #[must_use]
    pub fn function_call_in_progress(
        id: impl Into<String>,
        call_id: impl Into<String>,
        function_name: impl Into<String>,
    ) -> Self {
        Self::FunctionCall {
            id: id.into(),
            call_id: call_id.into(),
            name: function_name.into(),
            arguments: String::new(),
            status: OpenAiResponseItemStatus::InProgress,
        }
    }

    #[must_use]
    pub fn reasoning(id: impl Into<String>, reasoning_text: impl Into<String>) -> Self {
        Self::Reasoning {
            id: id.into(),
            summary: vec![OpenAiResponseReasoningSummary {
                summary_type: "summary_text",
                text: reasoning_text.into(),
            }],
            content: Vec::new(),
            status: OpenAiResponseItemStatus::Completed,
        }
    }

    #[must_use]
    pub fn message(id: impl Into<String>, output_text: impl Into<String>) -> Self {
        Self::Message {
            id: id.into(),
            role: "assistant",
            content: vec![OpenAiResponseOutputContent {
                content_type: "output_text",
                text: output_text.into(),
                annotations: Vec::new(),
                logprobs: Vec::new(),
            }],
            status: OpenAiResponseItemStatus::Completed,
        }
    }

    #[must_use]
    pub fn function_call(
        id: impl Into<String>,
        call_id: impl Into<String>,
        function_name: impl Into<String>,
        arguments_json: impl Into<String>,
    ) -> Self {
        Self::FunctionCall {
            id: id.into(),
            call_id: call_id.into(),
            name: function_name.into(),
            arguments: arguments_json.into(),
            status: OpenAiResponseItemStatus::Completed,
        }
    }

    fn message_text(&self) -> Option<&str> {
        match self {
            Self::Message { content, .. } => content.first().map(|part| part.text.as_str()),
            Self::Reasoning { .. } | Self::FunctionCall { .. } => None,
        }
    }

    fn mark_incomplete(&mut self) {
        match self {
            Self::Reasoning { status, .. }
            | Self::Message { status, .. }
            | Self::FunctionCall { status, .. } => {
                *status = OpenAiResponseItemStatus::Incomplete;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiResponseItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseReasoningSummary {
    #[serde(rename = "type")]
    summary_type: &'static str,
    text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseReasoningContent {
    #[serde(rename = "type")]
    content_type: &'static str,
    text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiResponseOutputContent {
    #[serde(rename = "type")]
    content_type: &'static str,
    text: String,
    annotations: Vec<serde_json::Value>,
    logprobs: Vec<serde_json::Value>,
}

impl OpenAiResponseOutputContent {
    #[must_use]
    pub fn output_text(output_text: impl Into<String>) -> Self {
        Self {
            content_type: "output_text",
            text: output_text.into(),
            annotations: Vec::new(),
            logprobs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseFunctionTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
    strict: bool,
}

impl OpenAiResponseFunctionTool {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        parameters: serde_json::Value,
        strict: bool,
    ) -> Self {
        Self {
            tool_type: "function",
            name: name.into(),
            description,
            parameters,
            strict,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseIncompleteDetails {
    reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseError {
    code: String,
    message: String,
}

/// Checked token accounting for one local response.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OpenAiResponseUsage {
    input_tokens: u32,
    input_tokens_details: OpenAiResponseInputTokenDetails,
    output_tokens: u32,
    output_tokens_details: OpenAiResponseOutputTokenDetails,
    total_tokens: u32,
}

impl OpenAiResponseUsage {
    pub fn new(
        input_tokens: u32,
        output_tokens: u32,
        cached_tokens: u32,
        reasoning_tokens: u32,
    ) -> Option<Self> {
        input_tokens
            .checked_add(output_tokens)
            .map(|total_tokens| Self {
                input_tokens,
                input_tokens_details: OpenAiResponseInputTokenDetails {
                    cache_write_tokens: 0,
                    cached_tokens,
                },
                output_tokens,
                output_tokens_details: OpenAiResponseOutputTokenDetails { reasoning_tokens },
                total_tokens,
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct OpenAiResponseInputTokenDetails {
    cache_write_tokens: u32,
    cached_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct OpenAiResponseOutputTokenDetails {
    reasoning_tokens: u32,
}
