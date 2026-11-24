//! Post-conversion evaluation: how far a finished quant's next-token
//! distribution drifts from a reference model's.
//!
//! The sweep's KL values rank modules one at a time; this measures the whole
//! artifact at once, which is the number that says whether the conversion
//! worked. It is the cheapest informative check there is (seconds on a small
//! model, minutes on a large one) and it catches the failure a size and a
//! load test cannot: a quant that loads, generates fluent text, and answers
//! a different question from the one asked.
//!
//! Ported from OptiQ's `evaluate_kl` (`optiq/eval/kl.py`).

pub mod kl_evaluation;
pub mod kl_statistics;
pub mod prompts;

pub use kl_evaluation::{KlDivergenceEngine, KlEvaluationRequest, evaluate_kl_divergence};
pub use kl_statistics::KlEvaluationSummary;
pub use prompts::{
    DEFAULT_EVALUATION_PROMPT_COUNT, DEFAULT_EVALUATION_SEQUENCE_LENGTH_TOKENS,
    bundled_evaluation_prompts, tokenized_evaluation_prompts,
};
