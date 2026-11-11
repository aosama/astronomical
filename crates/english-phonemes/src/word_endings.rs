//! The -s, -ed and -ing endings of words the lexicon lists only in their
//! stem form, with the sound each ending takes after the stem's last sound.

use crate::lexicon::{Lexicon, WordContext};

/// Vowels after which an American "t" becomes a flap ("butter", "waiting").
const FLAPPING_VOWELS: &str = "AIOWYiuæɑəɛɪɹʊʌ";

impl Lexicon {
    /// "cats" from "cat", "boxes" from "box", "stories" from "story".
    pub(crate) fn stem_s(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        if !word.is_ascii() || word.len() < 3 || !word.ends_with('s') {
            return None;
        }
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1]) {
            word[..word.len() - 1].to_owned()
        } else if (word.ends_with("'s")
            || (word.len() > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&word[..word.len() - 2])
        {
            word[..word.len() - 2].to_owned()
        } else if word.len() > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", &word[..word.len() - 3]))
        {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return None;
        };
        let stem_phonemes = self.lookup(&stem, stress_level, Some(context))?;
        Some(with_s_ending(&stem_phonemes))
    }

    /// "walked" from "walk", "wanted" from "want".
    pub(crate) fn stem_ed(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        if !word.is_ascii() || word.len() < 4 || !word.ends_with('d') {
            return None;
        }
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1]) {
            &word[..word.len() - 1]
        } else if word.len() > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&word[..word.len() - 2])
        {
            &word[..word.len() - 2]
        } else {
            return None;
        };
        let stem_phonemes = self.lookup(stem, stress_level, Some(context))?;
        Some(with_ed_ending(&stem_phonemes))
    }

    /// "reading" from "read", "making" from "make", "running" from "run".
    pub(crate) fn stem_ing(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: WordContext,
    ) -> Option<String> {
        if !word.is_ascii() || word.len() < 5 || !word.ends_with("ing") {
            return None;
        }
        let bare_stem = &word[..word.len() - 3];
        let stem = if word.len() > 5 && self.is_known(bare_stem) {
            bare_stem.to_owned()
        } else if self.is_known(&format!("{bare_stem}e")) {
            format!("{bare_stem}e")
        } else if word.len() > 5
            && has_doubled_final_consonant(word)
            && self.is_known(&word[..word.len() - 4])
        {
            word[..word.len() - 4].to_owned()
        } else {
            return None;
        };
        let stem_phonemes = self.lookup(&stem, stress_level, Some(context))?;
        Some(with_ing_ending(&stem_phonemes))
    }
}

/// Misaki's `([bcdgklmnprstvxz])\1ing$|cking$`.
fn has_doubled_final_consonant(word: &str) -> bool {
    if word.ends_with("cking") {
        return true;
    }
    let letters: Vec<char> = word.chars().collect();
    let letter_count = letters.len();
    letter_count >= 5
        && letters[letter_count - 4] == letters[letter_count - 5]
        && "bcdgklmnprstvxz".contains(letters[letter_count - 4])
}

pub(crate) fn with_s_ending(stem_phonemes: &str) -> String {
    match stem_phonemes.chars().last() {
        Some(last_sound) if "ptkfθ".contains(last_sound) => format!("{stem_phonemes}s"),
        Some(last_sound) if "szʃʒʧʤ".contains(last_sound) => format!("{stem_phonemes}ᵻz"),
        _ => format!("{stem_phonemes}z"),
    }
}

pub(crate) fn with_ed_ending(stem_phonemes: &str) -> String {
    let stem_sounds: Vec<char> = stem_phonemes.chars().collect();
    match stem_sounds.last() {
        Some(last_sound) if "pkfθʃsʧ".contains(*last_sound) => format!("{stem_phonemes}t"),
        Some('d') => format!("{stem_phonemes}ᵻd"),
        Some('t')
            if stem_sounds.len() >= 2
                && FLAPPING_VOWELS.contains(stem_sounds[stem_sounds.len() - 2]) =>
        {
            let without_t: String = stem_sounds[..stem_sounds.len() - 1].iter().collect();
            format!("{without_t}ɾᵻd")
        }
        Some('t') if stem_sounds.len() < 2 => format!("{stem_phonemes}ɪd"),
        Some('t') => format!("{stem_phonemes}ᵻd"),
        _ => format!("{stem_phonemes}d"),
    }
}

pub(crate) fn with_ing_ending(stem_phonemes: &str) -> String {
    let stem_sounds: Vec<char> = stem_phonemes.chars().collect();
    let ends_in_flappable_t = stem_sounds.len() > 1
        && stem_sounds[stem_sounds.len() - 1] == 't'
        && FLAPPING_VOWELS.contains(stem_sounds[stem_sounds.len() - 2]);
    if ends_in_flappable_t {
        let without_t: String = stem_sounds[..stem_sounds.len() - 1].iter().collect();
        return format!("{without_t}ɾɪŋ");
    }
    format!("{stem_phonemes}ɪŋ")
}

#[cfg(test)]
#[path = "word_endings_tests.rs"]
mod tests;
