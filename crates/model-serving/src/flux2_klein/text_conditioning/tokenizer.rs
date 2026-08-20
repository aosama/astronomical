//! Tokenizer JSON loading from the artifact's retained read-only descriptor.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;

use tokenizers::Tokenizer;

use super::batch::{Flux2KleinPreparedTextBatch, prepare_token_rows};
use super::error::Flux2KleinTextConditioningError;

const TOKENIZER_DESCRIPTOR_NAME: &str = "tokenizer/tokenizer.json";
const MAXIMUM_TOKENIZER_BYTES: u64 = 32 * 1024 * 1024;

pub(crate) struct Flux2KleinTokenizer {
    tokenizer: Tokenizer,
}

impl Flux2KleinTokenizer {
    pub(crate) fn from_retained_sidecars(
        retained_sidecars: &BTreeMap<String, File>,
    ) -> Result<Self, Flux2KleinTextConditioningError> {
        let tokenizer_file = retained_sidecars
            .get(TOKENIZER_DESCRIPTOR_NAME)
            .ok_or(Flux2KleinTextConditioningError::MissingTokenizerDescriptor)?;
        let tokenizer_bytes = read_bounded_descriptor_bytes(tokenizer_file)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|source| Flux2KleinTextConditioningError::TokenizerLoad { source })?;
        Ok(Self { tokenizer })
    }

    pub(crate) fn prepare_rendered_prompts(
        &self,
        rendered_prompts: &[String],
    ) -> Result<Flux2KleinPreparedTextBatch, Flux2KleinTextConditioningError> {
        if rendered_prompts.is_empty() {
            return Err(Flux2KleinTextConditioningError::EmptyPromptBatch);
        }
        let mut encoded_prompt_rows = Vec::with_capacity(rendered_prompts.len());
        for rendered_prompt in rendered_prompts {
            let encoding = self
                .tokenizer
                .encode(rendered_prompt.as_str(), false)
                .map_err(|source| Flux2KleinTextConditioningError::PromptTokenization { source })?;
            encoded_prompt_rows.push(encoding.get_ids().to_vec());
        }
        prepare_token_rows(encoded_prompt_rows)
    }
}

fn read_bounded_descriptor_bytes(
    tokenizer_file: &File,
) -> Result<Vec<u8>, Flux2KleinTextConditioningError> {
    let byte_count = tokenizer_file
        .metadata()
        .map_err(Flux2KleinTextConditioningError::TokenizerDescriptorIo)?
        .len();
    if byte_count > MAXIMUM_TOKENIZER_BYTES {
        return Err(Flux2KleinTextConditioningError::TokenizerDescriptorTooLarge);
    }
    let byte_count = usize::try_from(byte_count)
        .map_err(|_| Flux2KleinTextConditioningError::TokenizerDescriptorTooLarge)?;
    let mut tokenizer_bytes = vec![0_u8; byte_count];
    tokenizer_file
        .read_exact_at(&mut tokenizer_bytes, 0)
        .map_err(Flux2KleinTextConditioningError::TokenizerDescriptorIo)?;
    Ok(tokenizer_bytes)
}
