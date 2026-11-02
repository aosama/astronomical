//! numpy's `RandomState(seed).shuffle`, reproduced exactly.
//!
//! OptiQ shuffles the calibration sequence offsets with
//! `np.random.RandomState(42)`. Matching it bit for bit means a Rust sweep
//! and a Python sweep of the same mix calibrate on the same sequences in the
//! same order, so their sensitivity tables can be compared directly instead
//! of only through the allocations they lead to. The algorithm is small and
//! frozen: numpy keeps `RandomState` on its legacy stream forever.
//!
//! - The generator is MT19937 seeded by `init_genrand`, which is what
//!   `RandomState` does with an integer seed.
//! - The shuffle is Fisher–Yates from the last element down, drawing each
//!   index with numpy's legacy `random_interval`: mask a 32-bit draw to the
//!   smallest all-ones value covering the bound, and reject draws above it.

#[cfg(test)]
#[path = "seeded_shuffle_tests.rs"]
mod seeded_shuffle_tests;

const STATE_WORD_COUNT: usize = 624;
const MIDDLE_WORD_OFFSET: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The 32-bit Mersenne Twister.
pub struct MersenneTwister19937 {
    state_words: [u32; STATE_WORD_COUNT],
    next_word_index: usize,
}

impl MersenneTwister19937 {
    /// `init_genrand`: the seed is truncated to 32 bits, as numpy does.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut state_words = [0_u32; STATE_WORD_COUNT];
        state_words[0] = (seed & 0xffff_ffff) as u32;
        for word_index in 1..STATE_WORD_COUNT {
            let previous_word = state_words[word_index - 1];
            state_words[word_index] = 1_812_433_253_u32
                .wrapping_mul(previous_word ^ (previous_word >> 30))
                .wrapping_add(word_index as u32);
        }
        Self {
            state_words,
            next_word_index: STATE_WORD_COUNT,
        }
    }

    /// The next tempered 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.next_word_index >= STATE_WORD_COUNT {
            self.regenerate_state();
        }
        let mut tempered_word = self.state_words[self.next_word_index];
        self.next_word_index += 1;
        tempered_word ^= tempered_word >> 11;
        tempered_word ^= (tempered_word << 7) & 0x9d2c_5680;
        tempered_word ^= (tempered_word << 15) & 0xefc6_0000;
        tempered_word ^ (tempered_word >> 18)
    }

    /// numpy's legacy `random_interval(max)`: a uniform draw in
    /// `[0, inclusive_maximum]` by masked rejection.
    pub fn next_in_interval(&mut self, inclusive_maximum: u32) -> u32 {
        if inclusive_maximum == 0 {
            return 0;
        }
        let mut covering_mask = inclusive_maximum;
        for shift in [1, 2, 4, 8, 16] {
            covering_mask |= covering_mask >> shift;
        }
        loop {
            let candidate = self.next_u32() & covering_mask;
            if candidate <= inclusive_maximum {
                return candidate;
            }
        }
    }

    fn regenerate_state(&mut self) {
        for word_index in 0..STATE_WORD_COUNT {
            let next_index = (word_index + 1) % STATE_WORD_COUNT;
            let middle_index = (word_index + MIDDLE_WORD_OFFSET) % STATE_WORD_COUNT;
            let mixed_word = (self.state_words[word_index] & UPPER_MASK)
                | (self.state_words[next_index] & LOWER_MASK);
            let mut twisted_word = mixed_word >> 1;
            if mixed_word & 1 == 1 {
                twisted_word ^= MATRIX_A;
            }
            self.state_words[word_index] = self.state_words[middle_index] ^ twisted_word;
        }
        self.next_word_index = 0;
    }
}

/// Shuffles `items` exactly as `np.random.RandomState(seed).shuffle` shuffles
/// a list of the same length.
pub fn shuffle_like_numpy_random_state<Item>(items: &mut [Item], seed: u64) {
    let mut generator = MersenneTwister19937::from_seed(seed);
    for swap_index in (1..items.len()).rev() {
        let drawn_index = generator.next_in_interval(swap_index as u32) as usize;
        items.swap(swap_index, drawn_index);
    }
}
