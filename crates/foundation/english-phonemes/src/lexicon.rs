//! The Misaki American English lexicon: its gold and silver entries with
//! Misaki's case variants, and the lookups on them (a known word's spelling,
//! a letter-by-letter acronym, a symbol's word). How a word token is read
//! from these is `word_reading`'s job.
//!
//! Misaki also reads a part-of-speech tag from spaCy for each word. This port
//! has no tagger, so a word with several readings takes its `DEFAULT` entry.
//! The cost is the occasional heteronym ("I read it yesterday" says "reed").

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value;

use crate::stress::{PRIMARY_STRESS, SECONDARY_STRESS, apply_stress};

const GOLD_LEXICON_JSON: &str = include_str!("../lexicon/us_gold.json");
const SILVER_LEXICON_JSON: &str = include_str!("../lexicon/us_silver.json");

/// Symbols Misaki reads as words wherever they stand.
const SYMBOL_WORDS: [(&str, &str); 4] =
    [("%", "percent"), ("&", "and"), ("+", "plus"), ("@", "at")];

/// One lexicon entry: one spelling, or a spelling per part-of-speech tag
/// (`DEFAULT` always present; a tag may map to nothing).
#[derive(Debug)]
enum LexiconEntry {
    Single(String),
    ByTag(HashMap<String, Option<String>>),
}

/// What the next word says about this one: whether it starts with a vowel,
/// `None` when a pause or the end follows.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WordContext {
    pub(crate) next_starts_with_vowel: Option<bool>,
}

/// Gold (checked by hand) and silver (gap-filling) spellings, each grown with
/// the capitalised and lowercase variants Misaki adds.
pub(crate) struct Lexicon {
    gold_entries: HashMap<String, LexiconEntry>,
    silver_entries: HashMap<String, LexiconEntry>,
}

impl Lexicon {
    /// The lexicon, parsed once per process: about 180,000 entries that take
    /// a few hundred milliseconds to read.
    pub(crate) fn american() -> &'static Lexicon {
        static AMERICAN_LEXICON: OnceLock<Lexicon> = OnceLock::new();
        AMERICAN_LEXICON.get_or_init(|| Lexicon {
            gold_entries: grown_entries(GOLD_LEXICON_JSON),
            silver_entries: grown_entries(SILVER_LEXICON_JSON),
        })
    }

    /// Whether `word` is spelled out in the gold or silver lexicon itself.
    pub(crate) fn has_entry(&self, word: &str) -> bool {
        self.gold_entries.contains_key(word) || self.silver_entries.contains_key(word)
    }

    pub(crate) fn is_known(&self, word: &str) -> bool {
        if self.gold_entries.contains_key(word)
            || self.silver_entries.contains_key(word)
            || SYMBOL_WORDS.iter().any(|(symbol, _)| *symbol == word)
        {
            return true;
        }
        if word.is_empty() || !word.chars().all(|letter| letter.is_ascii_alphabetic()) {
            return false;
        }
        if word.len() == 1 {
            return true;
        }
        if word == word.to_uppercase() && self.gold_entries.contains_key(&word.to_lowercase()) {
            return true;
        }
        word[1..] == word[1..].to_uppercase()
    }

    /// Misaki's `lookup` with no part-of-speech tag.
    pub(crate) fn lookup(
        &self,
        word: &str,
        stress_level: Option<f32>,
        context: Option<WordContext>,
    ) -> Option<String> {
        let lookup_word = if word == word.to_uppercase() && !self.gold_entries.contains_key(word) {
            word.to_lowercase()
        } else {
            word.to_owned()
        };
        let entry = self
            .gold_entries
            .get(&lookup_word)
            .or_else(|| self.silver_entries.get(&lookup_word));
        let Some(spelling) = entry.and_then(|entry| entry_spelling(entry, context)) else {
            return self.spelled_letters(&lookup_word);
        };
        Some(apply_stress(&spelling, stress_level))
    }

    /// The default spelling of a gold entry, for the small words whose rules
    /// name one.
    pub(crate) fn gold_default(&self, word: &str) -> Option<String> {
        self.gold_entries
            .get(word)
            .and_then(|entry| entry_spelling(entry, None))
    }

    /// An acronym read letter by letter, the last letter carrying the stress.
    pub(crate) fn spelled_letters(&self, word: &str) -> Option<String> {
        let letters: Vec<char> = word
            .chars()
            .filter(|letter| letter.is_alphabetic())
            .collect();
        if letters.is_empty() {
            return None;
        }
        let mut letter_phonemes = String::new();
        for letter in letters {
            letter_phonemes.push_str(&self.gold_default(&letter.to_uppercase().to_string())?);
        }
        let demoted = apply_stress(&letter_phonemes, Some(0.0));
        Some(match demoted.rsplit_once(SECONDARY_STRESS) {
            Some((before_last, last_letter)) => {
                format!("{before_last}{PRIMARY_STRESS}{last_letter}")
            }
            None => demoted,
        })
    }

    /// The symbol's word ("%" is "percent"), when `word` is one.
    pub(crate) fn symbol_phonemes(&self, word: &str) -> Option<String> {
        let (_, symbol_word) = SYMBOL_WORDS.iter().find(|(symbol, _)| *symbol == word)?;
        self.lookup(symbol_word, None, None)
    }
}

fn entry_spelling(entry: &LexiconEntry, context: Option<WordContext>) -> Option<String> {
    match entry {
        LexiconEntry::Single(spelling) => Some(spelling.clone()),
        LexiconEntry::ByTag(spelling_by_tag) => {
            let before_a_pause =
                context.is_some_and(|context| context.next_starts_with_vowel.is_none());
            let tag = if before_a_pause && spelling_by_tag.contains_key("None") {
                "None"
            } else {
                "DEFAULT"
            };
            spelling_by_tag.get(tag).cloned().flatten()
        }
    }
}

/// Parses one lexicon file and adds Misaki's case variants: a lowercase entry
/// also answers for its Capitalised form and the other way round.
fn grown_entries(lexicon_json: &str) -> HashMap<String, LexiconEntry> {
    let parsed: serde_json::Map<String, Value> =
        serde_json::from_str(lexicon_json).expect("the vendored Misaki lexicon is valid JSON");
    let mut entries: HashMap<String, LexiconEntry> = HashMap::with_capacity(parsed.len() * 2);
    let mut case_variants: Vec<(String, String)> = Vec::new();
    for (word, spelling_value) in parsed {
        if word.chars().count() >= 2 {
            let lowercase_word = word.to_lowercase();
            let capitalised_word = capitalised(&word);
            if word == lowercase_word && word != capitalised_word {
                case_variants.push((capitalised_word, word.clone()));
            } else if word == capitalised(&lowercase_word) {
                case_variants.push((lowercase_word, word.clone()));
            }
        }
        if let Some(entry) = parsed_entry(&spelling_value) {
            entries.insert(word, entry);
        }
    }
    for (variant_word, source_word) in case_variants {
        if entries.contains_key(&variant_word) {
            continue;
        }
        if let Some(entry) = entries.get(&source_word).map(cloned_entry) {
            entries.insert(variant_word, entry);
        }
    }
    entries
}

fn parsed_entry(spelling_value: &Value) -> Option<LexiconEntry> {
    match spelling_value {
        Value::String(spelling) => Some(LexiconEntry::Single(spelling.clone())),
        Value::Object(spelling_by_tag) => Some(LexiconEntry::ByTag(
            spelling_by_tag
                .iter()
                .map(|(tag, spelling)| (tag.clone(), spelling.as_str().map(str::to_owned)))
                .collect(),
        )),
        _ => None,
    }
}

fn cloned_entry(entry: &LexiconEntry) -> LexiconEntry {
    match entry {
        LexiconEntry::Single(spelling) => LexiconEntry::Single(spelling.clone()),
        LexiconEntry::ByTag(spelling_by_tag) => LexiconEntry::ByTag(spelling_by_tag.clone()),
    }
}

/// Python's `str.capitalize`: first letter upper, the rest lower.
fn capitalised(word: &str) -> String {
    let mut letters = word.chars();
    match letters.next() {
        Some(first_letter) => first_letter
            .to_uppercase()
            .chain(letters.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

#[cfg(test)]
#[path = "lexicon_tests.rs"]
mod tests;
