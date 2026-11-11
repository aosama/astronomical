//! English numbers read aloud, as the Misaki pronunciation tool reads them:
//! which words a number token is said with, and where a number begins and
//! ends in running text. Nothing here knows about phonemes, so any speech
//! front end can use it.

mod number_spans;
mod number_words;
mod spoken_number;

pub use number_spans::{number_span_end, starts_signed_number};
pub use number_words::{cardinal_words, decimal_words, ordinal_words, year_words};
pub use spoken_number::{NumberEnding, SpokenNumber, is_number_token, spoken_number};
