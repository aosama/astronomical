//! The parts of Kokoro's `config.json` the forward pass depends on.
//!
//! Most of Kokoro's shape is fixed by the published checkpoint; reading the
//! values that vary between StyleTTS2 descendants, and refusing a file whose
//! values this port was not written for, turns a wrong artifact into a clear
//! load error instead of garbled audio.

use std::collections::HashMap;

use serde_json::Value;

use super::short_time_fourier::{FOURIER_SIZE, HOP_SIZE};
use super::waveform_generator::{
    RESIDUAL_DILATIONS, RESIDUAL_KERNEL_SIZES, UPSAMPLE_KERNEL_SIZES, UPSAMPLE_RATES,
};
use crate::speech_models::speech_model_error::SpeechModelError;

/// The checkpoint this port reproduces: Kokoro-82M v1.0.
const EXPECTED_HIDDEN_WIDTH: u64 = 512;
const EXPECTED_STYLE_WIDTH: u64 = 128;
const EXPECTED_PREDICTOR_LAYER_COUNT: u64 = 3;
const EXPECTED_ALBERT_HIDDEN_WIDTH: u64 = 768;
const EXPECTED_DURATION_BUCKET_COUNT: u64 = 50;
const EXPECTED_GENERATOR_INITIAL_CHANNELS: u64 = 512;

#[derive(Debug)]
pub(crate) struct KokoroConfiguration {
    /// Phoneme character to token id; characters outside it are not spoken.
    pub(crate) phoneme_vocabulary: HashMap<char, i32>,
    pub(crate) albert_attention_head_count: usize,
    pub(crate) albert_layer_count: usize,
}

impl KokoroConfiguration {
    pub(crate) fn parse(configuration_json: &str) -> Result<Self, SpeechModelError> {
        let document: Value = serde_json::from_str(configuration_json).map_err(|parse_error| {
            SpeechModelError::Artifact(format!("config.json is not JSON: {parse_error}"))
        })?;
        require_value(&document, "hidden_dim", EXPECTED_HIDDEN_WIDTH)?;
        require_value(&document, "style_dim", EXPECTED_STYLE_WIDTH)?;
        require_value(&document, "n_layer", EXPECTED_PREDICTOR_LAYER_COUNT)?;
        let albert = &document["plbert"];
        require_value(albert, "hidden_size", EXPECTED_ALBERT_HIDDEN_WIDTH)?;
        require_value(&document, "max_dur", EXPECTED_DURATION_BUCKET_COUNT)?;
        require_vocoder_shape(&document["istftnet"])?;
        Ok(Self {
            phoneme_vocabulary: phoneme_vocabulary(&document)?,
            albert_attention_head_count: positive_count(albert, "num_attention_heads")?,
            albert_layer_count: positive_count(albert, "num_hidden_layers")?,
        })
    }

    /// A configuration holding only a vocabulary, for tests of what reads it.
    #[cfg(test)]
    pub(crate) fn from_vocabulary_for_tests(vocabulary_entries: &[(char, i32)]) -> Self {
        Self {
            phoneme_vocabulary: vocabulary_entries.iter().copied().collect(),
            albert_attention_head_count: 12,
            albert_layer_count: 12,
        }
    }

    /// The token ids Kokoro reads for `phonemes`, between the start and end
    /// markers (id 0); a phoneme the vocabulary lacks is skipped, as Kokoro's
    /// own pipeline does.
    pub(crate) fn token_ids(&self, phonemes: &str) -> Vec<i32> {
        let mut token_ids = vec![0];
        token_ids.extend(
            phonemes
                .chars()
                .filter_map(|phoneme| self.phoneme_vocabulary.get(&phoneme).copied()),
        );
        token_ids.push(0);
        token_ids
    }
}

/// The vocoder's shape is written into the generator as constants; a
/// checkpoint with another shape is refused here rather than decoded wrong.
fn require_vocoder_shape(vocoder: &Value) -> Result<(), SpeechModelError> {
    require_value(vocoder, "gen_istft_n_fft", FOURIER_SIZE as u64)?;
    require_value(vocoder, "gen_istft_hop_size", HOP_SIZE as u64)?;
    require_value(
        vocoder,
        "upsample_initial_channel",
        EXPECTED_GENERATOR_INITIAL_CHANNELS,
    )?;
    require_integers(vocoder, "upsample_rates", &UPSAMPLE_RATES)?;
    require_integers(vocoder, "upsample_kernel_sizes", &UPSAMPLE_KERNEL_SIZES)?;
    require_integers(vocoder, "resblock_kernel_sizes", &RESIDUAL_KERNEL_SIZES)?;
    require_dilations(vocoder)
}

/// One dilation list per residual kernel size, each the same three dilations.
fn require_dilations(vocoder: &Value) -> Result<(), SpeechModelError> {
    let dilation_lists = vocoder["resblock_dilation_sizes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let every_list_matches = dilation_lists.len() == RESIDUAL_KERNEL_SIZES.len()
        && dilation_lists
            .iter()
            .all(|dilations| integers_of(dilations).as_deref() == Some(&RESIDUAL_DILATIONS[..]));
    if every_list_matches {
        return Ok(());
    }
    Err(SpeechModelError::Artifact(format!(
        "config.json istftnet resblock_dilation_sizes is {dilation_lists:?}; this engine \
         reproduces Kokoro-82M, where each of {} lists is {RESIDUAL_DILATIONS:?}",
        RESIDUAL_KERNEL_SIZES.len()
    )))
}

fn require_integers(section: &Value, key: &str, expected: &[i32]) -> Result<(), SpeechModelError> {
    let found = integers_of(&section[key]);
    if found.as_deref() == Some(expected) {
        return Ok(());
    }
    Err(SpeechModelError::Artifact(format!(
        "config.json {key} is {found:?}; this engine reproduces Kokoro-82M, where it is {expected:?}"
    )))
}

fn integers_of(list: &Value) -> Option<Vec<i32>> {
    list.as_array()?
        .iter()
        .map(|entry| {
            entry
                .as_i64()
                .and_then(|integer| i32::try_from(integer).ok())
        })
        .collect()
}

fn phoneme_vocabulary(document: &Value) -> Result<HashMap<char, i32>, SpeechModelError> {
    let vocabulary = document["vocab"]
        .as_object()
        .ok_or_else(|| SpeechModelError::Artifact("config.json has no vocab table".to_owned()))?;
    let mut phoneme_vocabulary = HashMap::with_capacity(vocabulary.len());
    for (phoneme_text, token_id) in vocabulary {
        let mut phoneme_characters = phoneme_text.chars();
        let (Some(phoneme), None) = (phoneme_characters.next(), phoneme_characters.next()) else {
            return Err(SpeechModelError::Artifact(format!(
                "vocab entry {phoneme_text:?} is not one character"
            )));
        };
        let token_id = token_id
            .as_i64()
            .and_then(|token_id| i32::try_from(token_id).ok())
            .ok_or_else(|| {
                SpeechModelError::Artifact(format!(
                    "vocab entry {phoneme_text:?} has no integer id"
                ))
            })?;
        phoneme_vocabulary.insert(phoneme, token_id);
    }
    Ok(phoneme_vocabulary)
}

/// ALBERT's head and layer counts must be at least 1: zero heads would
/// divide the width by zero.
fn positive_count(section: &Value, key: &str) -> Result<usize, SpeechModelError> {
    section[key]
        .as_u64()
        .filter(|found| *found > 0)
        .and_then(|found| usize::try_from(found).ok())
        .ok_or_else(|| {
            SpeechModelError::Artifact(format!("config.json has no positive count {key}"))
        })
}

fn require_value(section: &Value, key: &str, expected: u64) -> Result<(), SpeechModelError> {
    match section[key].as_u64() {
        Some(found) if found == expected => Ok(()),
        found => Err(SpeechModelError::Artifact(format!(
            "config.json {key} is {found:?}; this engine reproduces Kokoro-82M, where it is {expected}"
        ))),
    }
}

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod tests;
