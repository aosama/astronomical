//! What each transformer layer has seen: its keys and values, and the
//! absolute number of positions read so far.
//!
//! The two numbers differ for a windowed transformer: rotary angles continue
//! from the true position however many keys are kept, while the mask and
//! the attention only ever see the stored keys. Keeping them apart is what
//! lets the Mimi decoder drop keys that have left its window.

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::speech_models::speech_model_error::SpeechModelError;

/// Keys and values are `[heads, stored positions, head width]`.
pub(crate) struct AttentionHistory {
    keys: Vec<Option<MlxArray>>,
    values: Vec<Option<MlxArray>>,
    absolute_position_count: usize,
}

impl AttentionHistory {
    pub(crate) fn empty(layer_count: usize) -> Self {
        Self {
            keys: (0..layer_count).map(|_| None).collect(),
            values: (0..layer_count).map(|_| None).collect(),
            absolute_position_count: 0,
        }
    }

    /// How many positions have been read, the first new one's rotary position.
    pub(crate) fn absolute_position_count(&self) -> usize {
        self.absolute_position_count
    }

    /// How many keys a layer holds (every layer holds the same number).
    pub(crate) fn stored_count(&self) -> usize {
        self.keys
            .first()
            .and_then(Option::as_ref)
            .map_or(0, |stored_keys| stored_keys.shape()[1] as usize)
    }

    /// The layer's keys and values with `new_keys` and `new_values` appended.
    pub(crate) fn append(
        &mut self,
        runtime: &MlxRuntime,
        layer_index: usize,
        new_keys: MlxArray,
        new_values: MlxArray,
    ) -> Result<(MlxArray, MlxArray), SpeechModelError> {
        let keys = appended(runtime, &mut self.keys[layer_index], new_keys)?;
        let values = appended(runtime, &mut self.values[layer_index], new_values)?;
        Ok((keys, values))
    }

    /// Records that `new_count` positions were read, then keeps only the
    /// latest `kept_count` keys per layer when a window bounds what any later
    /// query may see.
    pub(crate) fn advance(
        &mut self,
        runtime: &MlxRuntime,
        new_count: usize,
        kept_count: Option<usize>,
    ) -> Result<(), SpeechModelError> {
        self.absolute_position_count += new_count;
        let Some(kept_count) = kept_count else {
            return Ok(());
        };
        for stored in self.keys.iter_mut().chain(self.values.iter_mut()) {
            *stored = stored
                .take()
                .map(|stored_entries| latest_positions(runtime, stored_entries, kept_count))
                .transpose()?;
        }
        Ok(())
    }

    /// An independent copy that can grow without changing this one.
    pub(crate) fn branch(&self) -> Result<Self, SpeechModelError> {
        Ok(Self {
            keys: retained_copies(&self.keys)?,
            values: retained_copies(&self.values)?,
            absolute_position_count: self.absolute_position_count,
        })
    }
}

fn appended(
    runtime: &MlxRuntime,
    stored: &mut Option<MlxArray>,
    new_entries: MlxArray,
) -> Result<MlxArray, SpeechModelError> {
    let combined = match stored.take() {
        Some(previous_entries) => {
            runtime.concatenate_axis(&[&previous_entries, &new_entries], 1)?
        }
        None => new_entries,
    };
    *stored = Some(combined.retain()?);
    Ok(combined)
}

fn latest_positions(
    runtime: &MlxRuntime,
    stored_entries: MlxArray,
    kept_count: usize,
) -> Result<MlxArray, SpeechModelError> {
    let [head_count, stored_count, head_width] = stored_entries.shape()[..] else {
        return Err(SpeechModelError::Input(
            "attention history expects [heads, positions, width]".to_owned(),
        ));
    };
    let kept_count = (kept_count as i32).min(stored_count);
    if kept_count == stored_count {
        return Ok(stored_entries);
    }
    Ok(runtime.slice(
        &stored_entries,
        &[0, stored_count - kept_count, 0],
        &[head_count, stored_count, head_width],
        &[1, 1, 1],
    )?)
}

fn retained_copies(stored: &[Option<MlxArray>]) -> Result<Vec<Option<MlxArray>>, SpeechModelError> {
    let mut copies = Vec::with_capacity(stored.len());
    for stored_entries in stored {
        copies.push(match stored_entries {
            Some(stored_entries) => Some(stored_entries.retain()?),
            None => None,
        });
    }
    Ok(copies)
}
