//! Row identity for the `qwen4_exp` n-gram embedding table (PLE).
//!
//! This owner is pure arithmetic: token identifiers in, logical row numbers
//! out. It owns the rule that turns a token window into row indices and
//! deliberately owns nothing else — shard mapping, byte ranges, caching, and
//! quantized row decoding belong to the artifact and storage layers, so the
//! rule can be pinned, audited, and reused no matter where the rows live.
//!
//! The rule is pinned from public reference implementations of this
//! architecture (see `docs/qwen4-exp-architecture.md`): sixteen hash heads,
//! per-head prime-sized tables, wrapping signed 64-bit multiply-XOR hashing,
//! and windows that never cross an end-of-sequence boundary. The contract
//! tests assert exact row indices for fixed token sequences, so a drifted
//! implementation fails loudly instead of producing fluent wrong text.

use std::fmt;

/// Splitmix64 additive constant used by the reference hash derivation.
pub const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// Splitmix64 first mixing multiplier.
pub const SPLITMIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Splitmix64 second mixing multiplier.
pub const SPLITMIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;
/// Per-layer seed prime the reference adds before mixing.
pub const PLE_LAYER_PRIME: u64 = 10_007;
/// Seed the reference assumes when configuration omits one.
pub const DEFAULT_NGRAM_SEED: u64 = 1234;

const I63_MAX: u64 = (1_u64 << 63) - 1;

/// Everything the row rule needs, resolved from validated configuration.
///
/// The seed and the PLE layer ordinal are resolved values: configuration may
/// omit the seed, and the ordinal is the zero-based position of the layer
/// among the PLE layers, not the layer index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramIdentityConfiguration {
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub unigram_vocab_size: u32,
    pub ngram_vocab_size_base: u64,
    pub vocabulary_divisor: u32,
    pub seed: u64,
    pub ple_layer_ordinal: u32,
    pub eos_token_id: u32,
}

/// Why an n-gram identity plan cannot be built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NgramPlanError {
    NgramSizeTooSmall { provided: u32 },
    ZeroHeadsPerNgram,
    ZeroUnigramVocabulary,
    VocabBaseTooSmall { provided: u64 },
    ZeroVocabularyDivisor,
}

impl fmt::Display for NgramPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NgramSizeTooSmall { provided } => {
                write!(formatter, "ngram_size must be at least 2, got {provided}")
            }
            Self::ZeroHeadsPerNgram => {
                write!(formatter, "heads_per_ngram must be positive")
            }
            Self::ZeroUnigramVocabulary => {
                write!(formatter, "unigram vocabulary size must be positive")
            }
            Self::VocabBaseTooSmall { provided } => write!(
                formatter,
                "ngram_vocab_size_base must be at least 2 so a prime above it exists, got {provided}"
            ),
            Self::ZeroVocabularyDivisor => {
                write!(
                    formatter,
                    "make_ngram_vocab_size_divisible_by must be positive"
                )
            }
        }
    }
}

impl std::error::Error for NgramPlanError {}

/// Why row indices cannot be computed for a token slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NgramIdentityError {
    ContextLengthMismatch { expected: usize, provided: usize },
}

impl fmt::Display for NgramIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextLengthMismatch { expected, provided } => write!(
                formatter,
                "n-gram context must hold exactly {expected} tokens, got {provided}"
            ),
        }
    }
}

impl std::error::Error for NgramIdentityError {}

/// The splitmix64 finalizer, in wrapping unsigned 64-bit arithmetic.
#[must_use]
pub fn splitmix64(value: u64) -> u64 {
    let mixed = value.wrapping_add(SPLITMIX_GAMMA);
    let mixed = (mixed ^ (mixed >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    mixed ^ (mixed >> 31)
}

/// Deterministic 64-bit primality through the Miller-Rabin witness set the
/// reference uses, so prime selection matches it exactly.
#[must_use]
pub fn is_prime_u64(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value % prime == 0 {
            return value == prime;
        }
    }
    let mut exponent = value - 1;
    let mut shifts = 0_u32;
    while exponent % 2 == 0 {
        exponent /= 2;
        shifts += 1;
    }
    for base in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if base % value == 0 {
            continue;
        }
        let mut witness = pow_mod_u64(base, exponent, value);
        if witness == 1 || witness == value - 1 {
            continue;
        }
        let mut witness_found = false;
        for _ in 1..shifts {
            witness = witness.wrapping_mul(witness) % value;
            if witness == value - 1 {
                witness_found = true;
                break;
            }
        }
        if !witness_found {
            return false;
        }
    }
    true
}

fn pow_mod_u64(base: u64, exponent: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    let mut base = base % modulus;
    let mut exponent = exponent;
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        base = base.wrapping_mul(base) % modulus;
        exponent /= 2;
    }
    result
}

/// The `count`-th prime strictly greater than `start`.
#[must_use]
pub fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        let mut candidate = prime + 1;
        if candidate <= 2 {
            prime = 2;
            continue;
        }
        if candidate % 2 == 0 {
            candidate += 1;
        }
        while !is_prime_u64(candidate) {
            candidate += 2;
        }
        prime = candidate;
    }
    prime
}

/// Per-head table sizes and offsets, plus the padded row count the published
/// manifests declare.
///
/// Offsets are cumulative over heads in ordinal order, so a row index is
/// unique across the whole table. The padded count rounds the total up to the
/// configured divisor; padding rows exist only as layout filler and are never
/// addressed by the hash.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramVocabLayout {
    head_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    unpadded_row_count: u64,
    padded_row_count: u64,
}

impl NgramVocabLayout {
    /// Builds the layout the reference derives: the head's global ordinal
    /// selects which prime after the base it owns.
    #[must_use]
    pub fn build(config: &NgramIdentityConfiguration) -> Self {
        let head_count = head_count_for(config.ngram_size, config.heads_per_ngram);
        let mut head_sizes = Vec::with_capacity(head_count);
        let mut head_offsets = Vec::with_capacity(head_count);
        let mut offset = 0_u64;
        for local_head in 0..head_count {
            let global_head =
                u64::from(config.ple_layer_ordinal) * head_count as u64 + local_head as u64;
            let size = nth_prime_after(config.ngram_vocab_size_base - 1, global_head + 1);
            head_sizes.push(size);
            head_offsets.push(offset);
            offset += size;
        }
        let divisor = u64::from(config.vocabulary_divisor);
        let padded = offset.div_ceil(divisor) * divisor;
        Self {
            head_sizes,
            head_offsets,
            unpadded_row_count: offset,
            padded_row_count: padded,
        }
    }

    /// Total hash heads across all n-gram orders.
    #[must_use]
    pub fn head_count(&self) -> usize {
        self.head_sizes.len()
    }

    /// Rows owned by one head.
    #[must_use]
    pub fn head_size(&self, head: usize) -> u64 {
        self.head_sizes[head]
    }

    /// Global row offset where one head's table begins.
    #[must_use]
    pub fn head_offset(&self, head: usize) -> u64 {
        self.head_offsets[head]
    }

    /// Rows addressable by the hash, before padding.
    #[must_use]
    pub fn unpadded_row_count(&self) -> u64 {
        self.unpadded_row_count
    }

    /// Rows the published storage declares, after divisor padding.
    #[must_use]
    pub fn padded_row_count(&self) -> u64 {
        self.padded_row_count
    }
}

/// Hash multipliers plus the resolved layout, ready to turn token windows
/// into row indices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramRowIdentity {
    layout: NgramVocabLayout,
    multipliers: Vec<u64>,
    ngram_size: u32,
    heads_per_ngram: u32,
    eos_token_id: u32,
}

impl NgramRowIdentity {
    /// Builds the identity from resolved configuration.
    ///
    /// # Errors
    /// When any configuration value cannot support the rule.
    pub fn build(config: &NgramIdentityConfiguration) -> Result<Self, NgramPlanError> {
        if config.ngram_size < 2 {
            return Err(NgramPlanError::NgramSizeTooSmall {
                provided: config.ngram_size,
            });
        }
        if config.heads_per_ngram == 0 {
            return Err(NgramPlanError::ZeroHeadsPerNgram);
        }
        if config.unigram_vocab_size == 0 {
            return Err(NgramPlanError::ZeroUnigramVocabulary);
        }
        if config.ngram_vocab_size_base < 2 {
            return Err(NgramPlanError::VocabBaseTooSmall {
                provided: config.ngram_vocab_size_base,
            });
        }
        if config.vocabulary_divisor == 0 {
            return Err(NgramPlanError::ZeroVocabularyDivisor);
        }
        // The bound keeps token * multiplier inside signed 63 bits, so the
        // multiply never overflows before the XOR mixes it.
        let half_bound = (I63_MAX / u64::from(config.unigram_vocab_size)) / 2;
        let half_bound = half_bound.max(1);
        let base_seed = config
            .seed
            .wrapping_add(PLE_LAYER_PRIME * u64::from(config.ple_layer_ordinal));
        let multipliers = (0..u64::from(config.ngram_size))
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        let layout = NgramVocabLayout::build(config);
        Ok(Self {
            layout,
            multipliers,
            ngram_size: config.ngram_size,
            heads_per_ngram: config.heads_per_ngram,
            eos_token_id: config.eos_token_id,
        })
    }

    /// The resolved vocabulary layout.
    #[must_use]
    pub fn layout(&self) -> &NgramVocabLayout {
        &self.layout
    }

    /// The derived per-position hash multipliers.
    #[must_use]
    pub fn multipliers(&self) -> &[u64] {
        &self.multipliers
    }

    /// Row indices for every token, `head_count` per token in head order.
    ///
    /// The context holds exactly `ngram_size - 1` tokens that precede
    /// `tokens`, end-of-sequence padded by the caller, so a chunk boundary
    /// never invents tokens that were never there.
    ///
    /// # Errors
    /// When the context length does not match the n-gram width.
    pub fn row_ids(
        &self,
        context: &[u32],
        tokens: &[u32],
    ) -> Result<Vec<Vec<u64>>, NgramIdentityError> {
        let width = self.ngram_size as usize - 1;
        if context.len() != width {
            return Err(NgramIdentityError::ContextLengthMismatch {
                expected: width,
                provided: context.len(),
            });
        }
        let mut combined = Vec::with_capacity(context.len() + tokens.len());
        combined.extend_from_slice(context);
        combined.extend_from_slice(tokens);
        // Distance since the last end-of-sequence strictly before each
        // position; a window slot farther back than this distance reads as
        // end-of-sequence, so a window never crosses a boundary.
        let mut segment = vec![0_i64; combined.len()];
        let mut last_eos: i64 = -1;
        for (position, token) in combined.iter().enumerate() {
            segment[position] = position as i64 - last_eos - 1;
            if *token == self.eos_token_id {
                last_eos = position as i64;
            }
        }
        let mut rows = Vec::with_capacity(tokens.len());
        for token_position in 0..tokens.len() {
            let column = token_position + width;
            let mut ids = Vec::with_capacity(self.layout.head_count());
            for ngram_order in 2..=self.ngram_size as usize {
                let mut mixed = 0_u64;
                for (offset, multiplier) in self.multipliers.iter().take(ngram_order).enumerate() {
                    let source = column as i64 - offset as i64;
                    let token = if source >= 0 && segment[source as usize] >= offset as i64 {
                        combined[source as usize]
                    } else {
                        self.eos_token_id
                    };
                    let product = u64::from(token) * multiplier;
                    mixed = if offset == 0 {
                        product
                    } else {
                        mixed ^ product
                    };
                }
                let head_start = (ngram_order - 2) * self.heads_per_ngram as usize;
                for head in head_start..head_start + self.heads_per_ngram as usize {
                    ids.push(mixed % self.layout.head_size(head) + self.layout.head_offset(head));
                }
            }
            rows.push(ids);
        }
        Ok(rows)
    }
}

/// Hash heads across all n-gram orders.
#[must_use]
pub fn head_count_for(ngram_size: u32, heads_per_ngram: u32) -> usize {
    usize::try_from(ngram_size.saturating_sub(1) * heads_per_ngram).expect("head count fits usize")
}
