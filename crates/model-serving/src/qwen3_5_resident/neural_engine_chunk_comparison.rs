//! Decides from a model's own prompts whether its programs make prefill faster, and
//! which engines each full chunk uses.
//!
//! The programs take half a chunk's MLP work off the GPU and add a hand-off per layer,
//! and which of those wins depends on the model: 1.28× on a 9B model, 1.03× to 1.11× on
//! a 2B model and 0.90× to 0.98× on a 0.8B model through the app, whatever the share (experiments.md).
//! Timing made-up tokens at load read 1.06, 0.68, 1.07 and 0.95 for one model, so the evidence comes
//! from real prompts instead: full chunks take turns between the engines, and each is
//! compared with the mean of its two neighbours, which cancels the steady growth in
//! chunk time as a prompt's position advances.

use std::time::{Duration, Instant};

/// Three prompts, because the first few seconds after the programs have been idle read
/// about 0.92 on a model that really gains 1.06.
const LEAST_COMPARED_PROMPTS: u32 = 3;
/// Single comparisons of one model vary by about 10 percent, and their median over this
/// many stayed within 3 percent of the long-run figure on all three models measured.
const LEAST_COMPARISONS: usize = 15;
/// Above 1.0 because the programs must repay more than this comparison sees: their wired
/// memory, and the wake-up after each idle spell, which took a 0.8B model reading 0.98 down
/// to 0.62 for a whole 20,000-token prompt. Models that gain measured 1.07 and up, and
/// wrongly turning off one this close to the line costs it only a few percent.
const LOWEST_WORTHWHILE_GAIN: f64 = 1.02;
/// The driver releases programs 10 to 20 seconds after their last use.
const PROGRAMS_GO_COLD_AFTER: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkEngines {
    Both,
    GpuAlone,
}

/// How the comparison came out. `gain` is how many times faster both engines ran than
/// the GPU alone, so below 1 means slower.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ChunkComparisonVerdict {
    Worthwhile { gain: f64 },
    NotWorthwhile { gain: f64 },
}

impl ChunkComparisonVerdict {
    pub(crate) fn from_gain(gain: f64) -> Self {
        if gain < LOWEST_WORTHWHILE_GAIN {
            Self::NotWorthwhile { gain }
        } else {
            Self::Worthwhile { gain }
        }
    }

    pub(crate) const fn gain(self) -> f64 {
        match self {
            Self::Worthwhile { gain } | Self::NotWorthwhile { gain } => gain,
        }
    }

    /// As the status carries it: 118 means 1.18 times.
    pub(crate) fn speed_percent(self) -> u32 {
        (self.gain() * 100.0).round() as u32
    }
}

/// Says why a model stays on the GPU. Gives the line it had to reach, because a reading
/// just under it would otherwise read as a gain being refused, and the only thing that
/// asks the question again, since nothing else will prompt a user to.
pub(crate) fn not_worthwhile_reason(gain: f64) -> String {
    format!(
        "the Neural Engine ran this model at {gain:.2} times the speed of the GPU alone, short of the {LOWEST_WORTHWHILE_GAIN:.2} it must reach, so prompts stay on the GPU; it is compared again after a macOS update"
    )
}

#[derive(Clone, Copy, Debug)]
struct TimedChunk {
    engines: ChunkEngines,
    end_token: usize,
    elapsed: Duration,
}

/// The engine choice for one loaded model's full chunks.
#[derive(Debug)]
pub(crate) struct ChunkComparison {
    full_chunk_token_count: usize,
    chosen_engines: ChunkEngines,
    next_engines_while_comparing: ChunkEngines,
    previous_chunks: [Option<TimedChunk>; 2],
    gains: Vec<f64>,
    compared_prompt_count: u32,
    has_compared_this_prompt: bool,
    verdict: Option<ChunkComparisonVerdict>,
    programs_last_used_at: Option<Instant>,
    led_with_gpu_at: Option<Instant>,
    did_lead_this_chunk: bool,
    is_prompt_warm: bool,
    last_chunk_end_token: Option<usize>,
}

impl ChunkComparison {
    pub(crate) const fn new(full_chunk_token_count: usize) -> Self {
        Self {
            full_chunk_token_count,
            chosen_engines: ChunkEngines::Both,
            next_engines_while_comparing: ChunkEngines::GpuAlone,
            previous_chunks: [None, None],
            gains: Vec::new(),
            compared_prompt_count: 0,
            has_compared_this_prompt: false,
            verdict: None,
            programs_last_used_at: None,
            led_with_gpu_at: None,
            did_lead_this_chunk: false,
            is_prompt_warm: true,
            last_chunk_end_token: None,
        }
    }

    /// One whose answer was found on an earlier load, so no prompt pays for finding it.
    pub(crate) const fn already_decided(
        full_chunk_token_count: usize,
        verdict: ChunkComparisonVerdict,
    ) -> Self {
        let mut comparison = Self::new(full_chunk_token_count);
        comparison.verdict = Some(verdict);
        comparison
    }

    /// Chooses the engines for a full chunk about to run.
    ///
    /// Programs that have gone cold lead with one chunk on the GPU alone: a cold first
    /// chunk on both engines cost 380 to 600 ms more than a warm one on all three models,
    /// and the same programs waking on the second chunk cost 140 to 180 ms.
    pub(crate) fn choose_engines(&mut self, now: Instant) -> ChunkEngines {
        self.did_lead_this_chunk = false;
        self.chosen_engines = match self.verdict {
            Some(ChunkComparisonVerdict::NotWorthwhile { .. }) => ChunkEngines::GpuAlone,
            _ if self.should_lead_with_gpu(now) => {
                self.did_lead_this_chunk = true;
                self.is_prompt_warm = false;
                ChunkEngines::GpuAlone
            }
            Some(ChunkComparisonVerdict::Worthwhile { .. }) => ChunkEngines::Both,
            None => self.next_engines_while_comparing,
        };
        self.chosen_engines
    }

    fn should_lead_with_gpu(&self, now: Instant) -> bool {
        has_gone_cold(self.programs_last_used_at, now) && has_gone_cold(self.led_with_gpu_at, now)
    }

    pub(crate) const fn verdict(&self) -> Option<ChunkComparisonVerdict> {
        self.verdict
    }

    /// Whether the chunk just chosen is a cold prompt's lead on the GPU.
    pub(crate) const fn did_lead_this_chunk(&self) -> bool {
        self.did_lead_this_chunk
    }

    /// Whether the current prompt began with the programs already awake.
    pub(crate) const fn is_prompt_warm(&self) -> bool {
        self.is_prompt_warm
    }

    /// A chunk that ran on engines this did not choose, such as the GPU alone because
    /// memory was short, says nothing about them and breaks the run of neighbours. It is
    /// still part of its prompt, which only a gap in the token positions can end.
    pub(crate) fn note_uncomparable_chunk(&mut self, start_token: usize, token_count: usize) {
        self.follow_prompt(start_token, token_count);
        self.previous_chunks = [None, None];
    }

    /// Keeps track of which prompt a finished chunk belongs to. A chunk that does not follow
    /// the last one starts another, which began warm unless this very chunk is its lead.
    fn follow_prompt(&mut self, start_token: usize, token_count: usize) {
        if self.last_chunk_end_token != Some(start_token) {
            self.has_compared_this_prompt = false;
            if !self.did_lead_this_chunk {
                self.is_prompt_warm = true;
            }
        }
        self.last_chunk_end_token = Some(start_token + token_count);
    }

    /// Takes one finished chunk. Only an unbroken run of full chunks is compared, because
    /// a shorter chunk's time says nothing about a full one's.
    pub(crate) fn note_finished_chunk(
        &mut self,
        start_token: usize,
        token_count: usize,
        elapsed: Duration,
        now: Instant,
    ) {
        if token_count != self.full_chunk_token_count {
            self.note_uncomparable_chunk(start_token, token_count);
            return;
        }
        let is_unbroken = self.last_chunk_end_token == Some(start_token);
        self.follow_prompt(start_token, token_count);
        if !is_unbroken {
            self.previous_chunks = [None, None];
        }
        // Stamped when the lead ends and not when it starts: a lead lasting longer than the
        // cold limit would otherwise look cold already and lead again, for ever, on a Mac
        // slow enough that one chunk takes that long.
        if self.did_lead_this_chunk {
            self.led_with_gpu_at = Some(now);
        }
        if self.chosen_engines == ChunkEngines::Both {
            self.programs_last_used_at = Some(now);
        }
        if self.verdict.is_some() {
            return;
        }
        let finished_chunk = TimedChunk {
            engines: self.chosen_engines,
            end_token: start_token + token_count,
            elapsed,
        };
        self.compare_with_neighbours(finished_chunk);
        self.previous_chunks = [self.previous_chunks[1], Some(finished_chunk)];
        self.next_engines_while_comparing = match finished_chunk.engines {
            ChunkEngines::Both => ChunkEngines::GpuAlone,
            ChunkEngines::GpuAlone => ChunkEngines::Both,
        };
        self.decide_when_enough();
    }

    fn compare_with_neighbours(&mut self, newest_chunk: TimedChunk) {
        let [Some(oldest_chunk), Some(middle_chunk)] = self.previous_chunks else {
            return;
        };
        let Some(gain) = gain_of_middle_chunk(oldest_chunk, middle_chunk, newest_chunk) else {
            return;
        };
        self.gains.push(gain);
        if !self.has_compared_this_prompt {
            self.has_compared_this_prompt = true;
            self.compared_prompt_count += 1;
        }
    }

    fn decide_when_enough(&mut self) {
        if self.gains.len() < LEAST_COMPARISONS
            || self.compared_prompt_count < LEAST_COMPARED_PROMPTS
        {
            return;
        }
        self.gains.sort_by(f64::total_cmp);
        let median_gain = self.gains[self.gains.len() / 2];
        self.verdict = Some(ChunkComparisonVerdict::from_gain(median_gain));
    }
}

/// Never having happened counts, so the first chunk after a load leads with the GPU too.
fn has_gone_cold(last_happened_at: Option<Instant>, now: Instant) -> bool {
    match last_happened_at {
        Some(last_happened_at) => {
            now.saturating_duration_since(last_happened_at) >= PROGRAMS_GO_COLD_AFTER
        }
        None => true,
    }
}

/// How much faster both engines ran, from one chunk against the mean of its neighbours.
/// Neighbours on the same engines as the middle compare nothing, which happens only when
/// a cold start repeats an arm.
fn gain_of_middle_chunk(
    oldest_chunk: TimedChunk,
    middle_chunk: TimedChunk,
    newest_chunk: TimedChunk,
) -> Option<f64> {
    if oldest_chunk.engines == middle_chunk.engines || newest_chunk.engines == middle_chunk.engines
    {
        return None;
    }
    let neighbours_seconds = f64::midpoint(
        oldest_chunk.elapsed.as_secs_f64(),
        newest_chunk.elapsed.as_secs_f64(),
    );
    let middle_seconds = middle_chunk.elapsed.as_secs_f64();
    if neighbours_seconds <= 0.0 || middle_seconds <= 0.0 {
        return None;
    }
    Some(match middle_chunk.engines {
        ChunkEngines::Both => neighbours_seconds / middle_seconds,
        ChunkEngines::GpuAlone => middle_seconds / neighbours_seconds,
    })
}

#[cfg(test)]
#[path = "neural_engine_chunk_comparison_tests.rs"]
mod tests;
