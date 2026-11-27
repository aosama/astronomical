//! A lexicon: every symbol a model knows (a token, or for Flash-Next an
//! n-gram) paired with the representation the model learned for it, stored
//! compressed as small integers with one scale and one offset per group of
//! numbers.
//!
//! Every reader of a compressed lexicon goes through
//! [`CompressedLexicon::learned_representations_of`]: the token lexicon held
//! in memory, a slice of it read from disk, and Flash-Next's n-gram lexicons.
//! Their representations are therefore the same operations by construction,
//! which is what keeps a lexicon left on disk bit-identical to one held in
//! memory.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

/// The three arrays of a compressed lexicon, borrowed from wherever they
/// live: the bound weights, a slice read from disk, or joined n-gram shards.
/// Row `n` of each array belongs to the lexicon's entry `n`.
pub struct CompressedLexicon<'arrays> {
    /// Each entry's numbers as small integers, packed into 32-bit words.
    pub packed_entries: &'arrays MlxArray,
    /// One scale for each group of an entry's numbers.
    pub entry_scales: &'arrays MlxArray,
    /// One offset for each group of an entry's numbers, added after the
    /// scale; MLX calls it the bias, a word that means something else in a
    /// network.
    pub entry_offsets: &'arrays MlxArray,
    pub bits_per_number: i32,
    pub numbers_per_group: i32,
}

impl CompressedLexicon<'_> {
    /// The learned representation of each of `selected_symbols`, unpacked to
    /// full numbers: one row per symbol, in the order given, a symbol asked
    /// for twice appearing twice.
    ///
    /// Only the selected entries are unpacked, never the whole lexicon: a
    /// Qwen 3.5 token lexicon holds about 248,000 entries and a prompt chunk
    /// asks for 2,048.
    ///
    /// # Errors
    /// When MLX refuses the arrays, for example scales that do not cover
    /// the packed entries' groups.
    pub fn learned_representations_of(
        &self,
        runtime: &MlxRuntime,
        selected_symbols: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(runtime.dequantize_affine(
            &runtime.take_axis(self.packed_entries, selected_symbols, 0)?,
            &runtime.take_axis(self.entry_scales, selected_symbols, 0)?,
            &runtime.take_axis(self.entry_offsets, selected_symbols, 0)?,
            self.numbers_per_group,
            self.bits_per_number,
        )?)
    }
}
