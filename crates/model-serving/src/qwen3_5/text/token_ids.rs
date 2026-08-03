use tokenizers::Tokenizer;

/// Token IDs and their corresponding string content from the Qwen3.5 vocabulary.
///
/// Each special token has both its numeric ID and its string representation so that
/// the tokenizer can validate bidirectional identity (ID -> content and content -> ID).
/// Token IDs are discovered from `tokenizer.json` rather than hardcoded per-model constants,
/// because the Qwen3.5 family shares the same special-token strings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3_5TokenIds {
    pub end_of_text_token_id: u32,
    pub end_of_text_token_content: &'static str,
    pub im_start_token_id: u32,
    pub im_start_token_content: &'static str,
    pub im_end_token_id: u32,
    pub im_end_token_content: &'static str,
    pub image_pad_token_id: u32,
    pub image_pad_token_content: &'static str,
    pub tool_call_start_token_id: u32,
    pub tool_call_start_token_content: &'static str,
    pub tool_call_end_token_id: u32,
    pub tool_call_end_token_content: &'static str,
    pub tool_response_start_token_id: u32,
    pub tool_response_start_token_content: &'static str,
    pub tool_response_end_token_id: u32,
    pub tool_response_end_token_content: &'static str,
    pub think_start_token_id: u32,
    pub think_start_token_content: &'static str,
    pub think_end_token_id: u32,
    pub think_end_token_content: &'static str,
}

impl Qwen3_5TokenIds {
    /// Returns an array of (content, id) pairs for all special tokens,
    /// in the order the tokenizer validates them.
    pub fn validation_pairs(&self) -> [(&'static str, u32); 10] {
        [
            (self.end_of_text_token_content, self.end_of_text_token_id),
            (self.im_start_token_content, self.im_start_token_id),
            (self.im_end_token_content, self.im_end_token_id),
            (self.image_pad_token_content, self.image_pad_token_id),
            (
                self.tool_call_start_token_content,
                self.tool_call_start_token_id,
            ),
            (
                self.tool_call_end_token_content,
                self.tool_call_end_token_id,
            ),
            (
                self.tool_response_start_token_content,
                self.tool_response_start_token_id,
            ),
            (
                self.tool_response_end_token_content,
                self.tool_response_end_token_id,
            ),
            (self.think_start_token_content, self.think_start_token_id),
            (self.think_end_token_content, self.think_end_token_id),
        ]
    }
}

const END_OF_TEXT_TOKEN_CONTENT: &str = "<|endoftext|>";
const IM_START_TOKEN_CONTENT: &str = "<|im_start|>";
const IM_END_TOKEN_CONTENT: &str = "<|im_end|>";
const IMAGE_PAD_TOKEN_CONTENT: &str = "<|image_pad|>";
const TOOL_CALL_START_TOKEN_CONTENT: &str = "<tool_call>";
const TOOL_CALL_END_TOKEN_CONTENT: &str = "</tool_call>";
const TOOL_RESPONSE_START_TOKEN_CONTENT: &str = "<tool_response>";
const TOOL_RESPONSE_END_TOKEN_CONTENT: &str = "</tool_response>";
const THINK_START_TOKEN_CONTENT: &str = "<think>";
const THINK_END_TOKEN_CONTENT: &str = "</think>";

/// Discovers the special token IDs from a loaded Qwen3.5-family tokenizer.
///
/// The Qwen3.5 family shares the same special token strings, but the numeric IDs
/// can differ between model snapshots. This function maps the known content strings
/// to their current IDs and returns them as a typed struct.
pub fn discover_token_ids(
    tokenizer: &Tokenizer,
) -> Result<Qwen3_5TokenIds, Qwen3_5TokenDiscoveryError> {
    let end_of_text_token_id = resolve_token_id(tokenizer, END_OF_TEXT_TOKEN_CONTENT)?;
    let im_start_token_id = resolve_token_id(tokenizer, IM_START_TOKEN_CONTENT)?;
    let im_end_token_id = resolve_token_id(tokenizer, IM_END_TOKEN_CONTENT)?;
    let image_pad_token_id = resolve_token_id(tokenizer, IMAGE_PAD_TOKEN_CONTENT)?;
    let tool_call_start_token_id = resolve_token_id(tokenizer, TOOL_CALL_START_TOKEN_CONTENT)?;
    let tool_call_end_token_id = resolve_token_id(tokenizer, TOOL_CALL_END_TOKEN_CONTENT)?;
    let tool_response_start_token_id =
        resolve_token_id(tokenizer, TOOL_RESPONSE_START_TOKEN_CONTENT)?;
    let tool_response_end_token_id = resolve_token_id(tokenizer, TOOL_RESPONSE_END_TOKEN_CONTENT)?;
    let think_start_token_id = resolve_token_id(tokenizer, THINK_START_TOKEN_CONTENT)?;
    let think_end_token_id = resolve_token_id(tokenizer, THINK_END_TOKEN_CONTENT)?;

    Ok(Qwen3_5TokenIds {
        end_of_text_token_id,
        end_of_text_token_content: END_OF_TEXT_TOKEN_CONTENT,
        im_start_token_id,
        im_start_token_content: IM_START_TOKEN_CONTENT,
        im_end_token_id,
        im_end_token_content: IM_END_TOKEN_CONTENT,
        image_pad_token_id,
        image_pad_token_content: IMAGE_PAD_TOKEN_CONTENT,
        tool_call_start_token_id,
        tool_call_start_token_content: TOOL_CALL_START_TOKEN_CONTENT,
        tool_call_end_token_id,
        tool_call_end_token_content: TOOL_CALL_END_TOKEN_CONTENT,
        tool_response_start_token_id,
        tool_response_start_token_content: TOOL_RESPONSE_START_TOKEN_CONTENT,
        tool_response_end_token_id,
        tool_response_end_token_content: TOOL_RESPONSE_END_TOKEN_CONTENT,
        think_start_token_id,
        think_start_token_content: THINK_START_TOKEN_CONTENT,
        think_end_token_id,
        think_end_token_content: THINK_END_TOKEN_CONTENT,
    })
}

fn resolve_token_id(
    tokenizer: &Tokenizer,
    token_content: &'static str,
) -> Result<u32, Qwen3_5TokenDiscoveryError> {
    let token_id = tokenizer
        .token_to_id(token_content)
        .ok_or(Qwen3_5TokenDiscoveryError::MissingSpecialToken { token_content })?;
    let round_trip_content = tokenizer
        .id_to_token(token_id)
        .ok_or(Qwen3_5TokenDiscoveryError::MissingSpecialToken { token_content })?;
    if round_trip_content != token_content {
        return Err(Qwen3_5TokenDiscoveryError::SpecialTokenIdentityMismatch {
            token_content,
            discovered_token_id: token_id,
            round_trip_content,
        });
    }
    Ok(token_id)
}

/// A failure while discovering special token IDs from a tokenizer.
#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5TokenDiscoveryError {
    #[error("special token '{token_content}' is missing from the tokenizer vocabulary")]
    MissingSpecialToken { token_content: &'static str },
    #[error(
        "special token '{token_content}' maps to id {discovered_token_id} but round-trips to '{round_trip_content}'"
    )]
    SpecialTokenIdentityMismatch {
        token_content: &'static str,
        discovered_token_id: u32,
        round_trip_content: String,
    },
}
