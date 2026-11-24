//! Phonon's own tokenizer (a SentencePiece unigram model in
//! `tokenizer.json`), read through the `tokenizers` crate.

use std::path::Path;

use tokenizers::Tokenizer;

use speech::speech_models::speech_model_error::SpeechModelError;

pub struct PhononTokenizer {
    tokenizer: Tokenizer,
}

impl PhononTokenizer {
    pub fn load(tokenizer_path: &Path) -> Result<Self, SpeechModelError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|load_error| {
            SpeechModelError::Artifact(format!("{}: {load_error}", tokenizer_path.display()))
        })?;
        Ok(Self { tokenizer })
    }

    /// Token ids without special tokens, as the reference runtime encodes.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, SpeechModelError> {
        let encoding = self.tokenizer.encode(text, false).map_err(|encode_error| {
            SpeechModelError::Input(format!("text could not be tokenized: {encode_error}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, SpeechModelError> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|decode_error| {
                SpeechModelError::Input(format!("tokens could not be decoded: {decode_error}"))
            })
    }
}
