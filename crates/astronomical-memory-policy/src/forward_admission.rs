//! The typed answer to whether the in-memory engine may run a forward.
//!
//! The in-memory engine keeps every weight, so it has nothing to give back
//! when a forward does not fit. Its only choices are to run the forward or
//! to say why the request has to move to the streaming engine.

/// Why a resident forward must leave the complete-residency engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingEngineRequirement {
    /// Persistent ownership alone exceeds the strict active-memory ceiling.
    StableShortfall { shortfall_bytes: usize },
    /// The configured prefill chunk's transient peak exceeds the ceiling.
    ///
    /// The chunk is never shrunk to make it fit: smaller chunks re-pay every
    /// per-chunk fixed cost (complete-layer decode and storage reads on the
    /// streaming side, publication and synchronization on both), so a
    /// halving ladder trades a clear engine decision for a slow request.
    TransientPeakShortfallAtPrefillChunk {
        shortfall_bytes: usize,
        prefill_chunk_token_count: usize,
    },
    /// A decode or other forward without a prefill chunk exceeds the peak allowance.
    TransientPeakShortfallWithoutPrefillChunk { shortfall_bytes: usize },
    /// The request's own context workspace cannot be admitted beside complete
    /// expert ownership. This is not a forward rejection: the forward never
    /// ran, and the SSD-streaming engine's smaller paged ownership may still
    /// admit the same context.
    RequestContextShortfall { shortfall_bytes: usize },
    /// A forward allocation exceeded the active-memory ceiling at runtime,
    /// after admission projected it to fit. The shortfall is exact.
    ForwardAllocationShortfall { shortfall_bytes: usize },
    /// A forward allocation exhausted graphics-processor memory at runtime.
    /// The failure is real but unquantified, so it carries no byte figure.
    ForwardGraphicsProcessorMemoryExhausted,
}

/// A concrete action after evaluating one resident forward's memory equation.
///
/// MLX buffer cache reclamation is intentionally absent: MLX active-memory
/// admission excludes MLX buffer cache bytes, and no resident forward admission
/// path reclaims the MLX buffer cache to make a projection fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentForwardAdmissionDecision {
    /// Stable ownership and expected peak fit.
    Admit,
    /// The resident engine cannot admit this forward without changing engines.
    RequireStreamingEngine { cause: StreamingEngineRequirement },
}

/// A resident forward decision that did not admit the forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentForwardAdmissionRejection {
    /// The resident engine cannot admit this forward without changing engines.
    RequireStreamingEngine { cause: StreamingEngineRequirement },
}

impl ResidentForwardAdmissionDecision {
    /// Chooses from what the memory ledger says the forward leaves short
    /// (issue #2221). `stable_shortfall_bytes` is how far the weights and
    /// the stored state after the forward exceed the limit by themselves;
    /// `peak_shortfall_bytes` adds the forward's MLX working buffers. A forward
    /// that does not fit is never made smaller: the request changes engine.
    #[must_use]
    pub fn for_measured_shortfalls(
        stable_shortfall_bytes: u64,
        peak_shortfall_bytes: u64,
        prefill_chunk_token_count: Option<usize>,
    ) -> Self {
        if stable_shortfall_bytes > 0 {
            return Self::RequireStreamingEngine {
                cause: StreamingEngineRequirement::StableShortfall {
                    shortfall_bytes: usize::try_from(stable_shortfall_bytes).unwrap_or(usize::MAX),
                },
            };
        }
        if peak_shortfall_bytes == 0 {
            return Self::Admit;
        }
        let shortfall_bytes = usize::try_from(peak_shortfall_bytes).unwrap_or(usize::MAX);
        let cause = match prefill_chunk_token_count {
            Some(prefill_chunk_token_count) => {
                StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                    shortfall_bytes,
                    prefill_chunk_token_count,
                }
            }
            None => StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes,
            },
        };
        Self::RequireStreamingEngine { cause }
    }

    /// Returns the rejection a caller must enact, or `None` when admitted.
    ///
    /// Rejection carriers hold this narrower type so no error path can carry
    /// `Admit` and no caller needs an unreachable "unexpected decision" arm.
    #[must_use]
    pub const fn rejection(self) -> Option<ResidentForwardAdmissionRejection> {
        match self {
            Self::Admit => None,
            Self::RequireStreamingEngine { cause } => {
                Some(ResidentForwardAdmissionRejection::RequireStreamingEngine { cause })
            }
        }
    }
}
