//! The four measured tasks and their prompts: writing an essay and writing
//! code, each after a short prompt and after one of about 10,000 tokens. The
//! long prompts are cut from the repository's Romeo and Juliet and MLX source
//! fixtures (AGENTS.md prefers Shakespeare for text; code needs code), and each
//! round asks about something different so no reply repeats another.

use std::path::Path;

/// Characters of the Romeo and Juliet fixture for about 10,000 tokens: English
/// runs about 3.7 characters a token in these tokenizers.
const LONG_ESSAY_CHARACTERS: usize = 37_000;
/// Characters of the MLX source fixture for about 10,000 tokens: Python source
/// packs about 4.5 characters a token.
const LONG_CODE_CHARACTERS: usize = 45_000;
/// The long prompts' usual size, which the character counts above give.
pub const USUAL_LONG_PROMPT_TOKENS: u32 = 10_000;
const FIXTURES_DIRECTORY: &str = "verification/acceptance/tests/fixtures";

const ESSAY_TOPICS: [&str; 3] = [
    "why people keep diaries",
    "what makes a city walkable",
    "how bees decide where to live",
];
const ESSAY_ASPECTS: [&str; 3] = [
    "the role of chance",
    "how the families talk to each other",
    "the pace of the ending",
];
const CODE_TASKS: [&str; 3] = [
    "parses a CSV file of expenses and prints a monthly summary",
    "implements a small least-recently-used lookup table with tests",
    "retries an HTTP request with exponential backoff and jitter",
];
const LONG_CODE_TASKS: [&str; 3] = [
    "computes the mean and variance of an array along an axis",
    "pads an array to a target shape with a constant value",
    "splits an array into equal chunks along an axis",
];

/// One measured task, in the order a round runs them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeedTask {
    ShortPromptEssay,
    ShortPromptCode,
    LongPromptEssay,
    LongPromptCode,
}

impl SpeedTask {
    pub const ALL: [Self; 4] = [
        Self::ShortPromptEssay,
        Self::ShortPromptCode,
        Self::LongPromptEssay,
        Self::LongPromptCode,
    ];

    /// The catalog's field name for this task's rate.
    #[must_use]
    pub const fn catalog_field(self) -> &'static str {
        match self {
            Self::ShortPromptEssay => "essay_short_prompt",
            Self::ShortPromptCode => "code_short_prompt",
            Self::LongPromptEssay => "essay_long_prompt",
            Self::LongPromptCode => "code_long_prompt",
        }
    }

    /// The temperature people use for the task, as the repository's
    /// measurement rules ask (the `desired_state_architecture` skill):
    /// 0.7 for writing code, 1.0 for general writing. Greedy is never a speed
    /// a person sees, and a drafter keeps more tokens at lower temperatures,
    /// so a lower one would overstate drafting.
    #[must_use]
    pub const fn default_temperature(self) -> f64 {
        match self {
            Self::ShortPromptCode | Self::LongPromptCode => 0.7,
            Self::ShortPromptEssay | Self::LongPromptEssay => 1.0,
        }
    }
}

/// The fixture texts the prompts are cut from, read once.
pub struct SpeedPrompts {
    romeo_and_juliet_text: String,
    source_code_text: String,
    warmup_text: String,
}

impl SpeedPrompts {
    pub fn load(repository_root: &Path) -> Result<Self, String> {
        Ok(Self {
            romeo_and_juliet_text: read_fixture(
                repository_root,
                "model_metrics_10000_tokens_romeo_and_juliet.txt",
            )?,
            source_code_text: read_fixture(
                repository_root,
                "model_metrics_progressive_source_code.txt",
            )?,
            warmup_text: read_fixture(
                repository_root,
                "model_metrics_warmup_romeo_and_juliet.txt",
            )?,
        })
    }

    /// About 500 prompt tokens, answered with 50: loads the model and warms
    /// its kernels before anything is measured.
    #[must_use]
    pub fn warmup_prompt(&self) -> String {
        format!("{}\n\nContinue the story above.", self.warmup_text)
    }

    /// A long prompt opens with its round's own instruction, so no round
    /// shares a long prefix with another round or with the warm-up: an engine
    /// that keeps a prompt it has already read would otherwise read rounds
    /// two and three at a speed no person's first prompt sees.
    #[must_use]
    pub fn task_prompt(
        &self,
        speed_task: SpeedTask,
        round_index: usize,
        long_prompt_tokens: u32,
    ) -> String {
        let round_slot = round_index % 3;
        match speed_task {
            SpeedTask::ShortPromptEssay => format!(
                "Write an essay of about 400 words on {}.",
                ESSAY_TOPICS[round_slot]
            ),
            SpeedTask::ShortPromptCode => {
                format!("Write a Python module that {}.", CODE_TASKS[round_slot])
            }
            SpeedTask::LongPromptEssay => format!(
                "Read the story below, then write an essay of about 400 words on it, focusing on {}.\n\n{}",
                ESSAY_ASPECTS[round_slot],
                leading_characters(
                    &self.romeo_and_juliet_text,
                    characters_for(LONG_ESSAY_CHARACTERS, long_prompt_tokens)
                )
            ),
            SpeedTask::LongPromptCode => format!(
                "Read the code below, then in the same style write a new function that {}.\n\n{}",
                LONG_CODE_TASKS[round_slot],
                leading_characters(
                    &self.source_code_text,
                    characters_for(LONG_CODE_CHARACTERS, long_prompt_tokens)
                )
            ),
        }
    }
}

fn read_fixture(repository_root: &Path, file_name: &str) -> Result<String, String> {
    let fixture_path = repository_root.join(FIXTURES_DIRECTORY).join(file_name);
    std::fs::read_to_string(&fixture_path)
        .map_err(|read_error| format!("could not read {}: {read_error}", fixture_path.display()))
}

/// The first characters of a text, cut on a character boundary.
/// The characters of a fixture for `long_prompt_tokens`, scaled from the
/// count that gives the usual 10,000.
fn characters_for(usual_character_count: usize, long_prompt_tokens: u32) -> usize {
    usual_character_count * long_prompt_tokens as usize / USUAL_LONG_PROMPT_TOKENS as usize
}

fn leading_characters(text: &str, character_count: usize) -> &str {
    text.char_indices()
        .nth(character_count)
        .map_or(text, |(byte_index, _)| &text[..byte_index])
}

#[cfg(test)]
#[path = "speed_prompts_tests.rs"]
mod speed_prompts_tests;
