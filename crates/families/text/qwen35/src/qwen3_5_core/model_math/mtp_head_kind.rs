//! The two kinds of multi-token-prediction head the engine drafts with.
//!
//! Both guess the token after next from the model's own state and both feed
//! the model's output head, but they read different state: Flash-Next's head
//! reads the model's several residual lanes and runs a Flash-Next layer with
//! routed experts; a dense Qwen 3.5 head reads the model's one final hidden
//! state and runs a plain full-attention layer. An enum keeps every place
//! that must tell them apart an exhaustive match.

use crate::qwen3_5_core::dense::mtp_head_weights::DenseMtpHeadWeights;
use crate::qwen4_exp::mtp_head_weights::FlashNextMtpHeadWeights;

#[derive(Debug)]
pub enum MtpHead {
    /// Shipped inside a Flash-Next checkpoint as its `mtp.*` tensors.
    FlashNext(Box<FlashNextMtpHeadWeights>),
    /// Shipped beside a dense Qwen 3.5 pack in its `mtp/` folder.
    DenseQwen3_5(Box<DenseMtpHeadWeights>),
}

/// Which kind of head a model hosts, apart from its weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpHeadKind {
    FlashNext,
    DenseQwen3_5,
}

impl MtpHeadKind {
    /// The draft count a reply uses when the configuration leaves it to
    /// the engine.
    ///
    /// Flash-Next drafts one token a round: streamed after a 10,000-token
    /// prompt it read 26.2 tokens a second against 20.5 with none at a 34 GB
    /// limit and 22.5 against 21.8 at 26 GB, where two drafts read 18.8 and
    /// three 17.8, because each further draft is one more row of experts to
    /// read and compute whether it is kept or not.
    ///
    /// A dense pack drafts none: Ternary Bonsai 2 9B, held whole, already
    /// writes about 80 tokens a second, and its head cost more than it saved
    /// (essay at temperature 1.0: 76.9 plain, 76.7 with one draft, 64.9 with
    /// two; code at 0.7: 70.8 plain, 66.9 with one). A one-draft round took
    /// 20.3 ms (2.9 drafting, 16.5 verifying two rows, 0.8 picking) for 1.56
    /// tokens, against 12.1 ms of GPU time for a plain token. The head still
    /// drafts when the configuration asks for a count.
    #[must_use]
    pub const fn automatic_draft_tokens(self) -> u32 {
        match self {
            Self::FlashNext => 1,
            Self::DenseQwen3_5 => 0,
        }
    }
}

/// The most tokens a round drafts: none without a hosted head, the
/// configured count when there is one, otherwise the head kind's own.
#[must_use]
pub fn resolved_mtp_draft_token_count(
    configured_draft_tokens: Option<u32>,
    hosted_head_kind: Option<MtpHeadKind>,
) -> u32 {
    let Some(head_kind) = hosted_head_kind else {
        return 0;
    };
    configured_draft_tokens.unwrap_or(head_kind.automatic_draft_tokens())
}

impl MtpHead {
    /// What the head occupies once a reply drafts with it.
    #[must_use]
    pub const fn payload_byte_count(&self) -> u64 {
        match self {
            Self::FlashNext(head) => head.payload_byte_count,
            Self::DenseQwen3_5(head) => head.payload_byte_count,
        }
    }

    /// Which kind of head this is.
    #[must_use]
    pub const fn kind(&self) -> MtpHeadKind {
        match self {
            Self::FlashNext(_) => MtpHeadKind::FlashNext,
            Self::DenseQwen3_5(_) => MtpHeadKind::DenseQwen3_5,
        }
    }
}

#[cfg(test)]
#[path = "mtp_head_kind_tests.rs"]
mod mtp_head_kind_tests;
