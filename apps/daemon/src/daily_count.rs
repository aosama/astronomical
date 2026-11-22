//! The daily count: one line a day from a Mac whose owner agreed, so the
//! developers know how many Macs use Astronomical and which versions they
//! run (issue #2419).
//!
//! The line holds four fixed values and no identifier of any kind, so a Mac
//! cannot be followed from one day to the next. Nothing is sent before the
//! person answers the one question about it, after "No thanks", or after the
//! switch is turned off.

mod daily_count_line;
mod daily_count_state;
mod daily_counter;

pub(crate) use daily_count_line::DailyCountLine;
pub(crate) use daily_count_state::{DailyCountState, unix_day_now};
pub(crate) use daily_counter::DailyCounter;
pub use daily_counter::keep_daily_count;
