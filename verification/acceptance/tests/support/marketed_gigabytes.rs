//! The one gigabyte unit acceptance journeys may use for `maximum_mlx_memory_gb`.
//!
//! The daemon expands every whole-gigabyte ceiling input by 2^30 bytes, on the
//! REST endpoint and in the config file alike. Journeys that converted with
//! 10^9 configured a ceiling about 7% above the bytes they then asserted
//! against, so their memory checks compared two different ceilings (issues
//! #1514, #1517). Every journey converts through here instead.

pub(crate) const BYTES_PER_MARKETED_GIGABYTE: u64 = 1 << 30;

/// Whole marketed gigabytes that never exceed `byte_count`: for a ceiling that
/// must stay below a target, such as a forced-streaming cell.
pub(crate) const fn marketed_gigabytes_at_most(byte_count: u64) -> u64 {
    byte_count / BYTES_PER_MARKETED_GIGABYTE
}

/// Whole marketed gigabytes that cover at least `byte_count`: for a ceiling
/// that must leave the journey its full intended budget.
pub(crate) const fn marketed_gigabytes_at_least(byte_count: u64) -> u64 {
    byte_count.div_ceil(BYTES_PER_MARKETED_GIGABYTE)
}

#[cfg(test)]
mod tests {
    use super::{
        BYTES_PER_MARKETED_GIGABYTE, marketed_gigabytes_at_least, marketed_gigabytes_at_most,
    };

    #[test]
    fn should_keep_an_exact_marketed_gigabyte_count_both_ways() {
        let twenty_gigabytes = 20 * BYTES_PER_MARKETED_GIGABYTE;
        assert_eq!(marketed_gigabytes_at_most(twenty_gigabytes), 20);
        assert_eq!(marketed_gigabytes_at_least(twenty_gigabytes), 20);
    }

    #[test]
    fn should_round_one_byte_over_down_for_at_most_and_up_for_at_least() {
        let one_byte_over = 20 * BYTES_PER_MARKETED_GIGABYTE + 1;
        assert_eq!(marketed_gigabytes_at_most(one_byte_over), 20);
        assert_eq!(marketed_gigabytes_at_least(one_byte_over), 21);
    }

    #[test]
    fn should_not_treat_a_decimal_gigabyte_count_as_marketed() {
        assert_eq!(marketed_gigabytes_at_most(40_000_000_000), 37);
    }

    #[test]
    fn should_return_zero_for_zero_bytes() {
        assert_eq!(marketed_gigabytes_at_most(0), 0);
        assert_eq!(marketed_gigabytes_at_least(0), 0);
    }
}
