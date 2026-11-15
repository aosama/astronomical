//! The expert cache as arithmetic: what is held, what a token needs, what
//! gets read from storage and what is given up to make room.
//!
//! A sparse model routes every token to a handful of experts in every layer.
//! When the model does not fit in memory, the experts that are not held have
//! to be read from storage, so the whole game is to hold the experts the
//! conversation actually uses. That is an ordinary caching problem, and this
//! module treats it as one: a single cache across all layers, one capacity,
//! one eviction order, and a record of every decision so a slow reply can be
//! explained from data instead of guessed at.
//!
//! Nothing here touches a model or a GPU. The engine mirrors the decisions
//! onto its weight tables, and a recorded reply can be replayed through the
//! same code on every commit (`recorded_reply_replay_tests`).
//!
//! - `routed_reply` — which experts each layer chose for each token of a
//!   real reply, and the compact file such a recording is kept in.
//! - `cache` — the cache itself.
//! - `cache_record` — what the cache did, per layer, per expert and per
//!   eviction.

mod cache;
mod cache_record;
mod routed_reply;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod recorded_reply_replay_tests;
#[cfg(test)]
mod routed_reply_tests;

pub use cache::{
    EvictionOrder, ExpertCache, ExpertCacheLayerGeometry, ExpertKey, LayerSlotLimit, ServedRoute,
};
pub use cache_record::{EvictionEvent, ExpertCacheRecord, ExpertUse, LayerCacheSummary};
pub use routed_reply::{RoutedReply, RoutedReplyDecodeError};
