//! Stress marks on a word's phonemes, ported from Misaki's `apply_stress`.
//!
//! Misaki expresses how strongly a word should be stressed as one number, and
//! Kokoro learned its prosody from phonemes adjusted by exactly these rules,
//! so the numbers and their thresholds are kept as Misaki wrote them:
//! below -1 strips every mark, -1 (or 0 and -0.5 on a word that has primary
//! stress) demotes primary to secondary, 0 to 1 adds secondary stress to an
//! unmarked word, 1 and above promotes secondary to primary, and above 1 adds
//! primary stress to an unmarked word.

pub(crate) const PRIMARY_STRESS: char = 'ˈ';
pub(crate) const SECONDARY_STRESS: char = 'ˌ';

/// Phonemes that carry a syllable, so a stress mark belongs in front of them.
pub(crate) const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";

pub(crate) fn is_vowel(phoneme: char) -> bool {
    VOWELS.contains(phoneme)
}

fn is_stress_mark(phoneme: char) -> bool {
    phoneme == PRIMARY_STRESS || phoneme == SECONDARY_STRESS
}

/// Applies Misaki's stress adjustment `stress_level` to `phonemes`; `None`
/// leaves the word as the lexicon spelled it.
pub(crate) fn apply_stress(phonemes: &str, stress_level: Option<f32>) -> String {
    let Some(stress_level) = stress_level else {
        return phonemes.to_owned();
    };
    let has_primary = phonemes.contains(PRIMARY_STRESS);
    let has_secondary = phonemes.contains(SECONDARY_STRESS);
    let has_any_stress = has_primary || has_secondary;
    if stress_level < -1.0 {
        return phonemes
            .chars()
            .filter(|phoneme| !is_stress_mark(*phoneme))
            .collect();
    }
    if stress_level == -1.0 || ((stress_level == 0.0 || stress_level == -0.5) && has_primary) {
        return phonemes
            .chars()
            .filter(|phoneme| *phoneme != SECONDARY_STRESS)
            .map(|phoneme| {
                if phoneme == PRIMARY_STRESS {
                    SECONDARY_STRESS
                } else {
                    phoneme
                }
            })
            .collect();
    }
    if [0.0, 0.5, 1.0].contains(&stress_level) && !has_any_stress {
        return stress_unmarked_word(phonemes, SECONDARY_STRESS);
    }
    if stress_level >= 1.0 && !has_primary && has_secondary {
        return phonemes.replace(SECONDARY_STRESS, &PRIMARY_STRESS.to_string());
    }
    if stress_level > 1.0 && !has_any_stress {
        return stress_unmarked_word(phonemes, PRIMARY_STRESS);
    }
    phonemes.to_owned()
}

fn stress_unmarked_word(phonemes: &str, stress_mark: char) -> String {
    if !phonemes.chars().any(is_vowel) {
        return phonemes.to_owned();
    }
    move_stress_marks_before_vowels(&format!("{stress_mark}{phonemes}"))
}

/// Moves every stress mark to just before the first vowel at or after it,
/// which is where Misaki's lexicon writes stress.
fn move_stress_marks_before_vowels(phonemes: &str) -> String {
    let phoneme_list: Vec<char> = phonemes.chars().collect();
    // Twice the sort position, so "just before vowel j" is the integer 2j - 1.
    let mut keyed_phonemes: Vec<(usize, char)> = Vec::with_capacity(phoneme_list.len());
    for (phoneme_index, phoneme) in phoneme_list.iter().enumerate() {
        let sort_key = if is_stress_mark(*phoneme) {
            phoneme_list[phoneme_index..]
                .iter()
                .position(|candidate| is_vowel(*candidate))
                .map_or(2 * phoneme_index, |offset| 2 * (phoneme_index + offset) - 1)
        } else {
            2 * phoneme_index
        };
        keyed_phonemes.push((sort_key, *phoneme));
    }
    keyed_phonemes.sort_by_key(|(sort_key, _)| *sort_key);
    keyed_phonemes
        .into_iter()
        .map(|(_, phoneme)| phoneme)
        .collect()
}

#[cfg(test)]
#[path = "stress_tests.rs"]
mod tests;
