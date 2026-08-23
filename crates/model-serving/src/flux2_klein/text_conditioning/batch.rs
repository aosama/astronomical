//! Pure right-padding and right-truncation geometry for fixed FLUX conditioning.

use super::error::Flux2KleinTextConditioningError;

pub(crate) const FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH: usize = 512;
pub(crate) const FLUX2_KLEIN_PAD_TOKEN_ID: u32 = 151_643;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Flux2KleinPreparedTextBatch {
    token_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    batch_size: usize,
}

impl Flux2KleinPreparedTextBatch {
    pub(crate) const fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub(crate) fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }
    pub(crate) const fn sequence_length(&self) -> usize {
        FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH
    }
}

pub(crate) fn prepare_token_rows(
    encoded_prompt_rows: Vec<Vec<u32>>,
) -> Result<Flux2KleinPreparedTextBatch, Flux2KleinTextConditioningError> {
    if encoded_prompt_rows.is_empty() {
        return Err(Flux2KleinTextConditioningError::EmptyPromptBatch);
    }
    let batch_size = encoded_prompt_rows.len();
    let element_count = batch_size
        .checked_mul(FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH)
        .ok_or(Flux2KleinTextConditioningError::BatchGeometryOverflow)?;
    let mut token_ids = Vec::with_capacity(element_count);
    let mut attention_mask = Vec::with_capacity(element_count);
    for encoded_prompt in encoded_prompt_rows {
        let retained_token_count = encoded_prompt
            .len()
            .min(FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH);
        token_ids.extend_from_slice(&encoded_prompt[..retained_token_count]);
        attention_mask.extend(std::iter::repeat_n(1, retained_token_count));
        let padding_token_count = FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH - retained_token_count;
        token_ids.extend(std::iter::repeat_n(
            FLUX2_KLEIN_PAD_TOKEN_ID,
            padding_token_count,
        ));
        attention_mask.extend(std::iter::repeat_n(0, padding_token_count));
    }
    Ok(Flux2KleinPreparedTextBatch {
        token_ids,
        attention_mask,
        batch_size,
    })
}
