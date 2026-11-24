//! Which questions a calibration or training was judged on.
//!
//! Judging on questions about texts the fit never saw is what tells a real
//! improvement from memory. With too few different texts there are none to
//! hold back, and the comparison falls back to the examples the fit learned
//! from; the result is still written, and everything downstream (the report,
//! the folder's training record, every answer the folder gives) says it was
//! not judged.

/// Which questions the comparison with the original model ran on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JudgedOn {
    /// Questions about texts the fit never saw.
    HeldBackQuestions,
    /// The examples the fit learned from, because none could be held back.
    LearnedFromExamples,
}
