//! The astro expert cache as arithmetic: what is held, what a token needs, what
//! gets read from storage and what is given up to make room.
//!
//! A sparse model routes every token to a handful of experts in every layer.
//! When the model does not fit in memory, the experts that are not held have
//! to be read from storage, so the whole game is to hold the experts the
//! conversation actually uses. That is an ordinary eviction-policy problem, and this
//! module treats it as one: a single astro expert cache across all layers, one capacity,
//! one eviction order, and a record of every decision so a slow reply can be
//! explained from data instead of guessed at.
//!
//! Nothing here touches a model or a GPU, so a recorded reply can be
//! replayed through this code on every commit
//! (`recorded_reply_replay_tests`). The engine does not follow
//! these decisions yet: its tables are still laid out by its own rules, and
//! this module is the measured design it is to be moved onto.
//!
//! - `routed_reply` — which experts each layer chose for each token of a
//!   real reply, and the compact file such a recording is kept in.
//! - `astro_expert_cache` — the astro expert cache with memory perfectly divisible, as a bound to
//!   compare against.
//! - `expert_tables` — the astro expert cache as an engine can hold it, one table per
//!   layer, with every decision stated as a change to a table.
//! - `astro_expert_cache_record` — what the astro expert cache did, per layer, per expert and per
//!   eviction.

mod astro_expert_cache;
mod astro_expert_cache_record;
mod expert_tables;
mod keeping_rules;
mod rebalanced_replay;
mod replay_case_file;
mod replay_lab;
mod routed_reply;

#[cfg(test)]
mod astro_expert_cache_tests;
#[cfg(test)]
mod expert_tables_tests;
#[cfg(test)]
mod rebalanced_replay_tests;
#[cfg(test)]
mod recorded_reply_replay_tests;
#[cfg(test)]
mod replay_case_file_tests;
#[cfg(test)]
mod replay_lab_tests;
#[cfg(test)]
mod routed_reply_tests;

pub use astro_expert_cache::{
    AstroExpertCacheLayerGeometry, DivisibleAstroExpertCache, EvictionOrder, ExpertKey,
    LayerSlotLimit, ServedRoute,
};
pub use astro_expert_cache_record::{
    AstroExpertCacheRecord, EvictionEvent, ExpertUse, LayerAstroExpertCacheSummary,
};
pub use expert_tables::{AstroExpertCacheTableRules, AstroExpertCacheTables, TableChange};
pub use keeping_rules::{
    BeladyMinimum, CappedReplacements, DecayedFrequency, FrequencyAdmittedRecency, KeepingRule,
    LeastFrequentlyUsed, LeastRecentlyUsed, LimitedLookaheadBelady, PriorSeededDecayedFrequency,
    ReplayPosition, SecondMostRecentUse,
};
pub use rebalanced_replay::replay_with_rebalanced_tables;
pub use replay_case_file::{
    ReplayCaseFileError, SavedReplayCase, find_saved_case_documents, read_saved_replay_case,
};
pub use replay_lab::{ReplayCase, ReplayOutcome, replay};
pub use routed_reply::{RoutedReply, RoutedReplyDecodeError};
