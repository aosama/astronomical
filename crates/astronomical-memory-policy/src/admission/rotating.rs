//! Architecture-neutral rotating-cache admission geometry.
//!
//! A sliding layer commits at most `window_size` tokens. Multi-token prefill
//! may temporarily own `window_size + prompt_chunk - 1` tokens so every new
//! token still sees a full prior window. Memory admission must charge that
//! transient, not only the steady-state ring.

/// Errors from rotating-cache admission arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RotatingAdmissionError {
    /// Window size was zero.
    ZeroWindowSize,
    /// Window plus chunk overflowed the destination integer.
    TransientTokenCountOverflow {
        window_size: u32,
        prompt_chunk_token_count: u32,
    },
}

/// Returns the peak token count owned during one rotating prefill chunk.
pub fn rotating_prefill_transient_token_count(
    window_size: u32,
    prompt_chunk_token_count: u32,
) -> Result<u32, RotatingAdmissionError> {
    if window_size == 0 {
        return Err(RotatingAdmissionError::ZeroWindowSize);
    }
    if prompt_chunk_token_count == 0 {
        return Ok(window_size);
    }
    window_size
        .checked_add(prompt_chunk_token_count)
        .and_then(|window_plus_chunk| window_plus_chunk.checked_sub(1))
        .ok_or(RotatingAdmissionError::TransientTokenCountOverflow {
            window_size,
            prompt_chunk_token_count,
        })
}

/// Returns how many tokens remain after a rotating update commits.
#[must_use]
pub fn rotating_committed_token_count(window_size: u32, absolute_position: u32) -> u32 {
    absolute_position.min(window_size)
}
