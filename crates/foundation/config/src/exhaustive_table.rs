//! A per-variant lookup written as a table.
//!
//! Some functions do nothing but give every variant of an enum its own
//! value: a model family its wire label, a catalog role its name. Written as
//! a `match`, each new variant makes such a function one branch larger, and
//! the repository's complexity ratchet (`cargo xtask crap-baseline`) reads
//! that as the function growing more complex, although a table with one more
//! row is no harder to read. This macro states the table as rows and expands
//! to the same exhaustive `match`, so a variant left out is still a compile
//! error.
//!
//! The protocol crate keeps its own copy because it sits below this crate
//! and must not depend on it.

/// Expands to an exhaustive `match` over `$subject`.
///
/// The first form gives each listed variant its own entry and ignores what
/// the variant holds. The second form binds what each listed tuple variant
/// holds to `$held` and evaluates the same `$entry` for all of them.
#[macro_export]
macro_rules! exhaustive_table {
    ($subject:expr, { $($variant:path => $entry:expr,)+ }) => {
        match $subject {
            $($variant { .. } => $entry,)+
        }
    };
    ($subject:expr, $held:ident => $entry:expr, [ $($variant:path),+ $(,)? ]) => {
        match $subject {
            $($variant($held) => $entry,)+
        }
    };
}
