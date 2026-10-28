//! Prefix-lookup contracts moved beside the pure lookup (issue #1610):
//! every test drives `PersistentPromptCachePrefixLookup::for_prompt` with
//! literal token vectors and predicate doubles, crossing no storage or
//! model boundary.

mod lookup_fixtures;
mod partial_tail_contracts;
mod prefix_lookup_contracts;
