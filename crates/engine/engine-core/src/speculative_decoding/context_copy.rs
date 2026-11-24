//! Context copies: when the text's last few tokens already appeared earlier,
//! propose what followed them then. Code, lists and quoted text repeat whole
//! runs, which a learned drafter limited to its block cannot reach.
//!
//! Ported from TensorFold's suffix lookup (release 1.2.0,
//! `zig/src/core/lanes/proposer.zig`, Apache-2.0, Copyright 2026 TensorFold
//! contributors): every earlier occurrence of the last `ngram` tokens is a
//! candidate, the one whose match runs furthest back wins (the most recent
//! on a tie), and it proposes only when that match is at least `min_match`
//! tokens. After `window` proposals in a row keep nothing it stays silent for
//! `silence_asks` asks, so text that does not repeat costs nothing.

/// TensorFold's settings for its serial Qwen 27B loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCopySettings {
    pub ngram: usize,
    pub minimum_match: usize,
    pub maximum_match: usize,
    pub window: usize,
    pub silence_asks: usize,
}

impl Default for ContextCopySettings {
    fn default() -> Self {
        Self {
            ngram: 3,
            minimum_match: 5,
            maximum_match: 64,
            window: 4,
            silence_asks: 16,
        }
    }
}

/// One request's copy proposer, with the memory of how its copies fared.
#[derive(Debug, Clone)]
pub struct ContextCopier {
    settings: ContextCopySettings,
    recent_kept_counts: Vec<usize>,
    silent_asks_left: usize,
}

impl ContextCopier {
    pub fn new(settings: ContextCopySettings) -> Self {
        Self {
            settings,
            recent_kept_counts: Vec::new(),
            silent_asks_left: 0,
        }
    }

    /// Up to `maximum_tokens` tokens that followed the best earlier match of
    /// `history`'s tail; empty when nothing matches well enough, or while the
    /// copier is silent.
    pub fn propose(&mut self, history: &[u32], maximum_tokens: usize) -> Vec<u32> {
        if maximum_tokens == 0 || history.len() <= self.settings.ngram {
            return Vec::new();
        }
        if self.silent_asks_left > 0 {
            self.silent_asks_left -= 1;
            return Vec::new();
        }
        let Some(continuation_start) = self.best_match_end(history) else {
            return Vec::new();
        };
        let copied_count = (history.len() - continuation_start).min(maximum_tokens);
        history[continuation_start..continuation_start + copied_count].to_vec()
    }

    /// Records how many of a proposal's tokens the round kept; a run of
    /// proposals that keep nothing silences the copier for a while.
    pub fn observe(&mut self, proposed_count: usize, kept_count: usize) {
        if proposed_count == 0 {
            return;
        }
        self.recent_kept_counts.push(kept_count);
        if self.recent_kept_counts.len() > self.settings.window {
            self.recent_kept_counts.remove(0);
        }
        let window_is_full = self.recent_kept_counts.len() >= self.settings.window;
        if window_is_full && self.recent_kept_counts.iter().all(is_zero) {
            self.silent_asks_left = self.settings.silence_asks;
            self.recent_kept_counts.clear();
        }
    }

    /// Where the continuation of the longest earlier match of the tail
    /// starts, if that match is long enough.
    fn best_match_end(&self, history: &[u32]) -> Option<usize> {
        let ngram = self.settings.ngram;
        let tail = &history[history.len() - ngram..];
        let mut best: Option<(usize, usize)> = None;
        // Most recent first: an earlier occurrence must match strictly
        // further back to win, as TensorFold's index walk does.
        for match_end in (ngram..history.len()).rev() {
            if &history[match_end - ngram..match_end] != tail {
                continue;
            }
            let match_length = self.match_length(history, match_end);
            let is_longer = match best {
                None => true,
                Some((_, best_length)) => match_length > best_length,
            };
            if is_longer {
                best = Some((match_end, match_length));
            }
        }
        match best {
            Some((match_end, match_length)) if match_length >= self.settings.minimum_match => {
                Some(match_end)
            }
            _ => None,
        }
    }

    /// How far back the text before `match_end` equals the text before the
    /// end of `history`, capped at `maximum_match`.
    fn match_length(&self, history: &[u32], match_end: usize) -> usize {
        let mut length = 0;
        while length < self.settings.maximum_match
            && length < match_end
            && history[match_end - 1 - length] == history[history.len() - 1 - length]
        {
            length += 1;
        }
        length
    }
}

fn is_zero(kept_count: &usize) -> bool {
    *kept_count == 0
}
