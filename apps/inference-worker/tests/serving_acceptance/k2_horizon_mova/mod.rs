//! User-journey acceptance for the K2 Horizon MoVA family.
//!
//! Journeys are sliced so a missing catalogue entry, unreadable answer, missed
//! tool call, blank memory panel, or zero tokens-per-second each fails alone.

mod advertise;
mod chat;
mod long_context_throughput;
mod memory;
mod responses;
mod support;
mod throughput;
mod tools;
