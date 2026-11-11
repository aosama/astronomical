//! Numbers spelled as the English words a speaker would say.
//!
//! Misaki spells numbers with the `num2words` package and then reads each
//! word from the lexicon, dropping "and" and splitting hyphenated words. These
//! functions return that word list directly, in the same order, so "2026"
//! reads as a year and "1,024" as a cardinal exactly as Kokoro heard them.

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 5] = [
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

/// "1234" as "one thousand two hundred thirty four".
pub fn cardinal_words(number: u64) -> Vec<String> {
    if number == 0 {
        return vec![ONES[0].to_owned()];
    }
    let mut spoken_words = Vec::new();
    let mut remainder = number;
    for (scale_value, scale_name) in SCALES {
        if remainder >= scale_value {
            spoken_words.extend(cardinal_words(remainder / scale_value));
            spoken_words.push(scale_name.to_owned());
            remainder %= scale_value;
        }
    }
    if remainder > 0 {
        spoken_words.extend(words_below_one_thousand(remainder));
    }
    spoken_words
}

fn words_below_one_thousand(number: u64) -> Vec<String> {
    let mut spoken_words = Vec::new();
    let hundreds = number / 100;
    let below_hundred = number % 100;
    if hundreds > 0 {
        spoken_words.push(ONES[hundreds as usize].to_owned());
        spoken_words.push("hundred".to_owned());
    }
    if below_hundred >= 20 {
        spoken_words.push(TENS[(below_hundred / 10) as usize].to_owned());
        if below_hundred % 10 > 0 {
            spoken_words.push(ONES[(below_hundred % 10) as usize].to_owned());
        }
    } else if below_hundred > 0 {
        spoken_words.push(ONES[below_hundred as usize].to_owned());
    }
    spoken_words
}

/// "21" as "twenty first": the cardinal with its last word made ordinal.
pub fn ordinal_words(number: u64) -> Vec<String> {
    let mut spoken_words = cardinal_words(number);
    if let Some(last_word) = spoken_words.last_mut() {
        *last_word = ordinal_of_word(last_word);
    }
    spoken_words
}

fn ordinal_of_word(cardinal_word: &str) -> String {
    match cardinal_word {
        "one" => "first".to_owned(),
        "two" => "second".to_owned(),
        "three" => "third".to_owned(),
        "five" => "fifth".to_owned(),
        "eight" => "eighth".to_owned(),
        "nine" => "ninth".to_owned(),
        "twelve" => "twelfth".to_owned(),
        word_ending_in_y if word_ending_in_y.ends_with('y') => {
            format!("{}ieth", &word_ending_in_y[..word_ending_in_y.len() - 1])
        }
        other_word => format!("{other_word}th"),
    }
}

/// "1984" as "nineteen eighty four" and "1905" as "nineteen oh five"; years
/// that people say as plain numbers ("2000", "2005", "1000") stay cardinal,
/// as `num2words` decides.
pub fn year_words(year: u64) -> Vec<String> {
    let century = year / 100;
    let year_in_century = year % 100;
    if century == 0 || (century % 10 == 0 && year_in_century < 10) || century >= 100 {
        return cardinal_words(year);
    }
    let mut spoken_words = cardinal_words(century);
    if year_in_century == 0 {
        spoken_words.push("hundred".to_owned());
    } else if year_in_century < 10 {
        spoken_words.push("oh".to_owned());
        spoken_words.extend(cardinal_words(year_in_century));
    } else {
        spoken_words.extend(cardinal_words(year_in_century));
    }
    spoken_words
}

/// "3.14" as "three point one four" and ".5" as "point five". `None` when the
/// text is not digits around one point.
pub fn decimal_words(decimal_text: &str) -> Option<Vec<String>> {
    let (whole_part, fraction_part) = decimal_text.split_once('.')?;
    if !is_all_digits(whole_part) || !is_all_digits(fraction_part) || fraction_part.is_empty() {
        return None;
    }
    let mut spoken_words = Vec::new();
    if !whole_part.is_empty() {
        spoken_words.extend(cardinal_words(whole_part.parse().ok()?));
    }
    spoken_words.push("point".to_owned());
    for digit in fraction_part.chars() {
        spoken_words.push(ONES[digit.to_digit(10)? as usize].to_owned());
    }
    Some(spoken_words)
}

fn is_all_digits(number_part: &str) -> bool {
    number_part
        .chars()
        .all(|character| character.is_ascii_digit())
}

#[cfg(test)]
#[path = "number_words_tests.rs"]
mod tests;
