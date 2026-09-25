//! Qwen-Image-2.1 text conditioning: template rendering + token-packaging pipeline.
//!
//! Replicates the **weights-free** contract of
//! `QwenImage21Pipeline._get_qwen_prompt_embeds` (diffusers reference) so the Rust engine
//! can produce the exact `(B, L)` id/mask matrices that feed the text-encoder forward:
//!
//!   1. normalize the user prompt (empty string becomes a single space, as the reference
//!      leaves the encoder nothing to read otherwise)
//!   2. render the raw `prompt_template_t2i` / `prompt_template_ti2i` string verbatim
//!      (NOT through `apply_chat_template` — the two tokenize differently and the
//!      checkpoint expects the raw-string form)
//!   3. tokenize with the artifact processor; valid positions come from the attention mask
//!   4. derive `drop_idx` as the length of the tokenized system message (14 for the
//!      reviewed artifact) and strip that many leading valid tokens per sample
//!   5. compute per-sample image-pad masks (`id == image_pad_token_id`)
//!   6. right-pad every sample to the post-drop batch maximum, stack into `(B, L)`
//!
//! The actual hidden-state *values* come from `text_encoder`, the GPU slice that consumes these
//! ids; this module owns the deterministic id/mask structure plus the template rendering that must
//! byte-match the reference, so it stays CPU-only and hermetically testable.
//!
//! # Scope caveats
//!
//! - In the reference, the vision processor expands each `<|image_pad|>` placeholder into
//!   one token per image patch when actual images are supplied. This module packages
//!   whatever ids arrive, so it is correct for expanded pads too; the expansion itself is
//!   a vision-processor concern for the image-conditioned slice.
//! - Padded `condition_ids` positions carry the placeholder id `0`. They are masked off by
//!   `encoder_attention_mask`; the text-encoder forward must consult that mask rather than
//!   treat id 0 as a vocabulary token.
//! - `drop_idx` is a parameter, not a constant: the reference derives it from
//!   `apply_chat_template` of the single system message, so the tokenizer-wiring slice must
//!   derive it the same way instead of hardcoding the reviewed artifact's value (14).
//!
//! Special-token note: the reference template embeds Qwen chat special tokens
//! (`<|im_start|>`, `<|im_end|>`, `<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`).
//! Those strings are built through [`special_tokens`] unicode escapes rather than raw
//! angle-bracket literals so no HTML-entity form (`&lt;`/`&gt;`) can ever leak into a
//! compiled template — the entity form tokenizes to completely different subword ids and
//! silently corrupts conditioning (this exact hazard stalled the previous onboarding
//! session; see the repo discovery guide).

use std::fmt;
use std::num::NonZeroUsize;

/// System prompt shared by every Qwen-Image-2.1 text-conditioning template.
pub const QWEN_IMAGE_21_SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Qwen chat special-token strings, bracketed with escaped angle characters.
///
/// `<` and `>` are expressed as unicode escapes on purpose: keeping the brackets
/// non-literal makes it impossible for an HTML-escaped `&lt;`/`&gt;` variant to compile
/// invisibly (it would tokenize to different ids and corrupt the whole conditioning path).
pub mod special_tokens {
    /// `<|im_start|>`
    pub const IM_START: &str = "\u{3c}|im_start|\u{3e}";
    /// `<|im_end|>`
    pub const IM_END: &str = "\u{3c}|im_end|\u{3e}";
    /// `<|image_pad|>`
    pub const IMAGE_PAD: &str = "\u{3c}|image_pad|\u{3e}";
    /// `<|vision_start|>`
    pub const VISION_START: &str = "\u{3c}|vision_start|\u{3e}";
    /// `<|vision_end|>`
    pub const VISION_END: &str = "\u{3c}|vision_end|\u{3e}";
}

use special_tokens::{IM_END, IM_START, IMAGE_PAD, VISION_END, VISION_START};

// --------------------------------------------------------------------------- //
// Template rendering                                                          //
// --------------------------------------------------------------------------- //

/// The reference replaces an empty prompt with a single space so the encoder always has
/// at least one token to read (`prompt = [" " if not p else p for p in prompt]`).
#[must_use]
pub fn normalize_empty_prompt(prompt: &str) -> &str {
    if prompt.is_empty() { " " } else { prompt }
}

fn reference_image_block(image_number: usize) -> String {
    format!("\u{3c}image{image_number}\u{3e}{VISION_START}{IMAGE_PAD}{VISION_END}")
}

/// Render the exact T2I prompt template verbatim. Matches the diffusers reference
/// `QwenImage21Pipeline.prompt_template_t2i` byte-for-byte.
#[must_use]
pub fn render_t2i_prompt_template(prompt: &str) -> String {
    format!(
        "{IM_START}system\n{SYS_PROMPT}{IM_END}\n\
         {IM_START}user\n{prompt}{IM_END}\n\
         {IM_START}assistant\n",
        SYS_PROMPT = QWEN_IMAGE_21_SYS_PROMPT,
        prompt = normalize_empty_prompt(prompt),
    )
}

/// Render the exact image-conditioned prompt template for a single reference image.
/// Matches `QwenImage21Pipeline.prompt_template_ti2i` with one image.
#[must_use]
pub fn render_ti2i_prompt_template(prompt: &str) -> String {
    render_ti2i_prompt_template_with_image_count(prompt, NonZeroUsize::MIN)
}

/// Render the image-conditioned prompt template for `image_count` reference images.
///
/// The reference inserts one `<imageN><|vision_start|><|image_pad|><|vision_end|>` block
/// per image into the user turn, separating the second and later blocks from the previous
/// one with a single space; the user prompt follows the last block.
#[must_use]
pub fn render_ti2i_prompt_template_with_image_count(
    prompt: &str,
    image_count: NonZeroUsize,
) -> String {
    let image_blocks: Vec<String> = (1..=image_count.get()).map(reference_image_block).collect();
    format!(
        "{IM_START}system\n{SYS_PROMPT}{IM_END}\n\
         {IM_START}user\n{image_blocks}{prompt}{IM_END}\n\
         {IM_START}assistant\n",
        SYS_PROMPT = QWEN_IMAGE_21_SYS_PROMPT,
        image_blocks = image_blocks.join(" "),
        prompt = normalize_empty_prompt(prompt),
    )
}

// --------------------------------------------------------------------------- //
// Packaging output                                                             //
// --------------------------------------------------------------------------- //

/// Packaged conditioning outputs: `(B, L_max)` matrices ready for the transformer.
///
/// These mirror the tensors the reference pipeline produces from
/// `_get_qwen_prompt_embeds` **after** applying `drop_idx`, before the text-encoder
/// embedding lookup turns ids into hidden states.
#[derive(Debug, Clone)]
pub struct PromptConditioningOutput {
    /// `(B, L_max)` ids for the text-encoder embedding table.
    pub condition_ids: Vec<Vec<u32>>,
    /// `(B, L_max)` attention mask: `1` where a valid token exists, `0` for padding.
    pub encoder_attention_mask: Vec<Vec<u8>>,
    /// `(B, L_max)` image-pad mask: `1` at positions holding an image-pad token, else `0`.
    pub image_pad_mask: Vec<Vec<u8>>,
}

impl PromptConditioningOutput {
    /// Batch dimension `B`.
    #[inline]
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.condition_ids.len()
    }

    /// Sequence dimension `L_max` (right-padded to uniform length). Returns `0` for an
    /// empty batch, where no padded row exists yet.
    #[inline]
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.condition_ids.first().map_or(0, |row| row.len())
    }
}

// --------------------------------------------------------------------------- //
// Error types                                                                  //
// --------------------------------------------------------------------------- //

/// Errors produced during the conditioning-packaging pipeline.
#[derive(Debug, Clone)]
pub enum TextConditioningError {
    /// The `input_ids` and `attention_mask` arrays have different batch lengths.
    InputMaskLengthMismatch,
    /// Within a sample the mask length does not match the id count.
    SampleLengthMismatch,
    /// `drop_idx` exceeds the number of valid tokens for one or more samples.
    DropIndexExceedsValidCount {
        /// Zero-based batch index of the offending sample.
        sample_index: usize,
        /// Number of valid (mask == 1) tokens found in that sample.
        valid_count: usize,
        /// The requested leading-token drop count.
        drop_idx: usize,
    },
}

impl fmt::Display for TextConditioningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputMaskLengthMismatch => write!(
                f,
                "input_ids and attention_mask arrays must have equal batch length"
            ),
            Self::SampleLengthMismatch => {
                write!(
                    f,
                    "per-sample input_ids and attention_mask must have equal length"
                )
            }
            Self::DropIndexExceedsValidCount {
                sample_index,
                valid_count,
                drop_idx,
            } => write!(
                f,
                "sample {sample_index}: drop_idx ({drop_idx}) exceeds valid-token count ({valid_count})"
            ),
        }
    }
}

impl std::error::Error for TextConditioningError {}

// --------------------------------------------------------------------------- //
// Core packaging function                                                      //
// --------------------------------------------------------------------------- //

/// Build packaged conditioning outputs from tokenized template ids and masks,
/// following the reference's `_extract_masked_hidden` / `drop_idx` / padding contract.
///
/// # Algorithm (mirrors the reference exactly)
///
/// 1. For each sample, extract valid positions where `attention_mask[j] > 0`
///    (this is `_extract_masked_hidden`: padding is dropped, whichever side it sat on).
/// 2. Drop the first `drop_idx` valid tokens from each sample (they belong to the
///    system role and carry no conditioning signal for the transformer).
/// 3. Compute `image_pad_mask` per sample: kept position → `1` when the id equals
///    `image_pad_token_id`, else `0`.
/// 4. Right-pad every sample to the post-drop batch maximum and stack into `(B, L_max)`
///    matrices; the encoder attention mask is all ones over kept positions and zero
///    over padding, exactly like the reference's `torch.ones` construction.
///
/// # Returns
///
/// `Ok(PromptConditioningOutput)` on success, or a typed error describing which
/// validation step failed.
pub fn build_prompt_conditioning(
    input_ids: &[Vec<u32>],
    attention_mask: &[Vec<u32>],
    drop_idx: usize,
    image_pad_token_id: u32,
) -> Result<PromptConditioningOutput, TextConditioningError> {
    if input_ids.len() != attention_mask.len() {
        return Err(TextConditioningError::InputMaskLengthMismatch);
    }

    if input_ids.is_empty() {
        return Ok(PromptConditioningOutput {
            condition_ids: Vec::new(),
            encoder_attention_mask: Vec::new(),
            image_pad_mask: Vec::new(),
        });
    }

    // Steps 1-2: extract valid tokens per sample, then drop the leading system block.
    let mut kept_ids: Vec<Vec<u32>> = Vec::with_capacity(input_ids.len());
    let mut kept_lengths: Vec<usize> = Vec::with_capacity(input_ids.len());

    for (sample_index, (ids, mask)) in input_ids.iter().zip(attention_mask.iter()).enumerate() {
        if ids.len() != mask.len() {
            return Err(TextConditioningError::SampleLengthMismatch);
        }

        let valid_ids: Vec<u32> = ids
            .iter()
            .zip(mask.iter())
            .filter_map(|(&id, &mask_value)| if mask_value > 0 { Some(id) } else { None })
            .collect();

        let valid_count = valid_ids.len();
        if drop_idx > valid_count {
            return Err(TextConditioningError::DropIndexExceedsValidCount {
                sample_index,
                valid_count,
                drop_idx,
            });
        }

        kept_ids.push(valid_ids[drop_idx..].to_vec());
        kept_lengths.push(valid_count - drop_idx);
    }

    // Step 4: post-drop batch maximum, then right-pad and stack all three outputs.
    let max_seq_len = kept_lengths.iter().copied().max().unwrap_or(0);

    let mut condition_ids = Vec::with_capacity(kept_ids.len());
    let mut encoder_attention_mask = Vec::with_capacity(kept_ids.len());
    let mut image_pad_mask = Vec::with_capacity(kept_ids.len());

    for (kept, &kept_length) in kept_ids.iter().zip(kept_lengths.iter()) {
        let mut sample_ids = kept.clone();
        sample_ids.resize(max_seq_len, 0);
        condition_ids.push(sample_ids);

        // Every kept position is valid by construction, so the mask is ones up to the
        // kept length and zeros over the right padding, matching torch.ones + stack.
        let mut sample_mask = vec![1u8; kept_length];
        sample_mask.resize(max_seq_len, 0);
        encoder_attention_mask.push(sample_mask);

        let mut sample_image_mask: Vec<u8> = kept
            .iter()
            .map(|&token_id| u8::from(token_id == image_pad_token_id))
            .collect();
        sample_image_mask.resize(max_seq_len, 0);
        image_pad_mask.push(sample_image_mask);
    }

    Ok(PromptConditioningOutput {
        condition_ids,
        encoder_attention_mask,
        image_pad_mask,
    })
}
