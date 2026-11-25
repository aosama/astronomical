//! How often MLX handed work between the caller, the GPU and its CPU stream.
//!
//! Every hand-off is a wait somewhere: a blocking evaluation makes the
//! caller wait for the GPU, a cross-stream wait makes the GPU wait for a
//! host step or the reverse. A reply token's speed depends on how many it
//! makes, so the counts are read from MLX's own evaluator and reported a
//! token, instead of being inferred from timings. A host readback of an
//! array nobody evaluated counts too: it evaluates inside MLX.

/// Totals since the process started. Subtract two readings for a span.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvaluationCounts {
    /// Evaluations that walked a graph and waited for it.
    pub blocking_evaluations: u64,
    /// Evaluations that walked a graph and returned without waiting.
    pub asynchronous_evaluations: u64,
    /// Waits for arrays an earlier evaluation had already been handed.
    pub waits_for_submitted_arrays: u64,
    /// Times one stream waited inside an evaluation for another's output:
    /// two for each host step, one in and one out.
    pub cross_stream_waits: u64,
}

impl EvaluationCounts {
    /// What happened between `earlier` and this reading. Counts only grow;
    /// a reading taken out of order gives zeros, not a wrapped total.
    #[must_use]
    pub const fn since(self, earlier: Self) -> Self {
        Self {
            blocking_evaluations: self
                .blocking_evaluations
                .saturating_sub(earlier.blocking_evaluations),
            asynchronous_evaluations: self
                .asynchronous_evaluations
                .saturating_sub(earlier.asynchronous_evaluations),
            waits_for_submitted_arrays: self
                .waits_for_submitted_arrays
                .saturating_sub(earlier.waits_for_submitted_arrays),
            cross_stream_waits: self
                .cross_stream_waits
                .saturating_sub(earlier.cross_stream_waits),
        }
    }
}

unsafe extern "C" {
    fn astro_evaluation_counts_read(counts: *mut EvaluationCounts);
}

/// The process's counts now.
#[must_use]
pub fn evaluation_counts() -> EvaluationCounts {
    let mut counts = EvaluationCounts::default();
    // SAFETY: the native side writes the four fields of the struct it is
    // handed, whose layout `repr(C)` matches field for field.
    unsafe { astro_evaluation_counts_read(&mut counts) };
    counts
}

#[cfg(test)]
#[path = "evaluation_counts_tests.rs"]
mod evaluation_counts_tests;
