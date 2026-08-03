const FIVE_THOUSAND_WORD_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const FIFTY_THOUSAND_WORD_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");
const HUNDRED_THOUSAND_WORD_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_100000_deterministic_words.txt");

pub(super) struct PersistentPromptCacheWarmupCase {
    pub(super) benchmark_name: &'static str,
    pub(super) instruction: &'static str,
    pub(super) maximum_output_tokens: u16,
    pub(super) source_document: &'static str,
    pub(super) source_word_count: usize,
}

pub(super) fn five_thousand_word_case() -> PersistentPromptCacheWarmupCase {
    PersistentPromptCacheWarmupCase {
        benchmark_name: "model_persistent_prompt_cache_warmup_5000_words",
        instruction: "Summarize the following document in exactly three concise paragraphs. Do not use headings or bullet points.",
        maximum_output_tokens: 2_000,
        source_document: FIVE_THOUSAND_WORD_FIXTURE,
        source_word_count: 5_000,
    }
}

pub(super) fn fifty_thousand_word_case() -> PersistentPromptCacheWarmupCase {
    PersistentPromptCacheWarmupCase {
        benchmark_name: "model_persistent_prompt_cache_warmup_50000_words",
        instruction: "Read the following public-domain book excerpt and reply with OK.",
        maximum_output_tokens: 1,
        source_document: FIFTY_THOUSAND_WORD_FIXTURE,
        source_word_count: 50_000,
    }
}

pub(super) fn hundred_thousand_word_case() -> PersistentPromptCacheWarmupCase {
    PersistentPromptCacheWarmupCase {
        benchmark_name: "model_persistent_prompt_cache_warmup_100000_words",
        instruction: "Read the following deterministic document and reply with OK.",
        maximum_output_tokens: 1,
        source_document: HUNDRED_THOUSAND_WORD_FIXTURE,
        source_word_count: 100_000,
    }
}
