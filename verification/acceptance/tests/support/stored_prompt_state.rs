//! A decoder state kept as the prompt left it, so a measurement can run its
//! reply tokens from the same starting point every pass.

use std::cell::RefCell;

use model_serving::{RequestDecoderStateStack, RequestDecoderStateStackCheckpoint};

/// The live decoder state and the checkpoint of it after the prompt.
pub struct StoredPromptState {
    pub decoder_state: RefCell<RequestDecoderStateStack>,
    stored_checkpoint: RefCell<Option<RequestDecoderStateStackCheckpoint>>,
}

impl StoredPromptState {
    /// Keeps `decoder_state`, which must hold the prompt and nothing after it.
    pub fn after_the_prompt(decoder_state: RequestDecoderStateStack) -> Self {
        Self {
            decoder_state: RefCell::new(decoder_state),
            stored_checkpoint: RefCell::new(None),
        }
    }

    /// Puts the decoder state back where the prompt left it, and keeps it so
    /// for the next pass. The first call only takes the checkpoint.
    pub fn restore(&self) -> Result<(), String> {
        let mut decoder_state = self.decoder_state.borrow_mut();
        let mut stored_checkpoint = self.stored_checkpoint.borrow_mut();
        if let Some(checkpoint) = stored_checkpoint.take() {
            decoder_state
                .restore_checkpoint(checkpoint)
                .map_err(|restore_error| restore_error.to_string())?;
        }
        *stored_checkpoint = Some(
            decoder_state
                .checkpoint()
                .map_err(|checkpoint_error| checkpoint_error.to_string())?,
        );
        Ok(())
    }
}
