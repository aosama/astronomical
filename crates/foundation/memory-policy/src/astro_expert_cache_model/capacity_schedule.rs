//! The capacity a recorded reply was really served under, token by token.
//!
//! The engine's `astro_expert_cache` capacity is the memory limit less every
//! need, and the needs move during a reply: the stored context grows, the
//! MLX working buffers grow when a larger step is first measured, and the
//! reserve that keeps macOS's wired total under its line follows every
//! reading of that total. A replay at one fixed capacity cannot show what
//! those moves cost, so the engine records a moment each time it applies a
//! capacity while a reply is recorded, and the lab replays the reply under
//! the same schedule.
//!
//! The reserve is recorded with what it was computed from, so a different
//! reserve rule can be replayed too: a token's capacity under another rule is
//! its recorded capacity, plus the reserve the engine held, less the reserve
//! the other rule would have held.

use serde::{Deserialize, Serialize};

/// One capacity the engine applied while a recorded reply was served.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapacityMoment {
    /// The reply token being served when the capacity was applied; the
    /// capacity holds from this token on.
    pub reply_token_index: u32,
    pub kept_expert_capacity_bytes: u64,
    /// The part of the limit kept unused for macOS's wired line.
    pub system_wired_reserve_bytes: u64,
    /// The system's wired total less what MLX held, at the latest reading.
    pub wired_outside_mlx_bytes: u64,
    /// When that reading was made, in milliseconds on the engine's monotonic
    /// clock; moments that share one reading share this.
    #[serde(default)]
    pub wired_reading_milliseconds: u64,
    /// The memory limit and the kernel's wire limit the reserve was computed
    /// against. A wire limit of zero means no wired reading had been made
    /// yet, so the reserve the engine held is the only one there is.
    pub limit_bytes: u64,
    pub system_wire_limit_bytes: u64,
    /// What the `astro_expert_cache` gave back because of this capacity.
    pub released_expert_bytes: u64,
}

/// The capacity each of `reply_token_count` tokens was served under. A
/// moment is applied after one token's forward and before the next, so token
/// `t` is served under the lowest of the moments recorded at index `t`, or,
/// when there are none, under the last capacity applied before it, starting
/// from `starting_capacity_bytes`.
///
/// The lowest, because a capacity that dips gives experts back then and
/// there, and one that rises again before the next token cannot bring them
/// back: the next token's misses read them again.
#[must_use]
pub fn capacity_by_token(
    capacity_moments: &[CapacityMoment],
    reply_token_count: usize,
    starting_capacity_bytes: u64,
) -> Vec<u64> {
    let mut moment_capacities = Vec::with_capacity(capacity_moments.len());
    for moment in capacity_moments {
        moment_capacities.push(moment.kept_expert_capacity_bytes);
    }
    schedule_by_token(
        capacity_moments,
        &moment_capacities,
        reply_token_count,
        starting_capacity_bytes,
    )
}

/// The same schedule as if the engine had held `reserve_by_moment[i]` for
/// macOS's wired line at moment `i` instead of what it held. A moment with no
/// entry in `reserve_by_moment` keeps the reserve the engine held.
#[must_use]
pub fn capacity_by_token_under_reserve(
    capacity_moments: &[CapacityMoment],
    reserve_by_moment: &[u64],
    reply_token_count: usize,
    starting_capacity_bytes: u64,
) -> Vec<u64> {
    let mut moment_capacities = Vec::with_capacity(capacity_moments.len());
    for (moment_index, moment) in capacity_moments.iter().enumerate() {
        let reserve_bytes = match reserve_by_moment.get(moment_index) {
            Some(&candidate_reserve_bytes) => candidate_reserve_bytes,
            None => moment.system_wired_reserve_bytes,
        };
        moment_capacities.push(
            moment
                .kept_expert_capacity_bytes
                .saturating_add(moment.system_wired_reserve_bytes)
                .saturating_sub(reserve_bytes),
        );
    }
    schedule_by_token(
        capacity_moments,
        &moment_capacities,
        reply_token_count,
        starting_capacity_bytes,
    )
}

fn schedule_by_token(
    capacity_moments: &[CapacityMoment],
    moment_capacities: &[u64],
    reply_token_count: usize,
    starting_capacity_bytes: u64,
) -> Vec<u64> {
    let mut capacity_bytes_by_token = Vec::with_capacity(reply_token_count);
    let mut carried_capacity_bytes = starting_capacity_bytes;
    let mut moment_index = 0;
    for token_index in 0..reply_token_count {
        let mut lowest_moment_capacity_bytes: Option<u64> = None;
        while let (Some(moment), Some(&capacity_bytes)) = (
            capacity_moments.get(moment_index),
            moment_capacities.get(moment_index),
        ) && moment.reply_token_index as usize <= token_index
        {
            lowest_moment_capacity_bytes = Some(match lowest_moment_capacity_bytes {
                Some(lowest_bytes) => lowest_bytes.min(capacity_bytes),
                None => capacity_bytes,
            });
            carried_capacity_bytes = capacity_bytes;
            moment_index += 1;
        }
        capacity_bytes_by_token
            .push(lowest_moment_capacity_bytes.unwrap_or(carried_capacity_bytes));
    }
    capacity_bytes_by_token
}

/// The moments of a reply whose first `dropped_token_count` tokens' routes
/// are no longer kept, indexed against the routes that are: the engine keeps
/// only a reply's newest routes, so a long reply's recording starts part way
/// through. A moment applied before the kept routes began is carried as one
/// at the first kept token, so the capacity those routes started under is
/// still known.
#[must_use]
pub fn rebased_capacity_moments(
    capacity_moments: &[CapacityMoment],
    dropped_token_count: u32,
) -> Vec<CapacityMoment> {
    let mut rebased_moments = Vec::with_capacity(capacity_moments.len());
    for moment in capacity_moments {
        let mut rebased_moment = *moment;
        rebased_moment.reply_token_index =
            moment.reply_token_index.saturating_sub(dropped_token_count);
        rebased_moments.push(rebased_moment);
    }
    rebased_moments
}

/// The reserve the ledger's rule holds against macOS's wired line for one
/// moment, had `wired_outside_mlx_bytes` been the memory wired outside MLX.
/// It is the ledger's own formula, so the lab cannot drift from the engine.
/// It leaves out the reserve the ledger grows while the system's wired total
/// is over the line, which the recordings this lab reads do not reach.
#[must_use]
pub const fn limit_past_wired_line_bytes(
    moment: &CapacityMoment,
    wired_outside_mlx_bytes: u64,
) -> u64 {
    crate::ledger::system_wired_ceiling::limit_past_system_wired_ceiling_bytes(
        moment.limit_bytes,
        crate::ledger::system_wired_ceiling::SystemWiredReading {
            system_wired_bytes: wired_outside_mlx_bytes,
            system_wire_limit_bytes: moment.system_wire_limit_bytes,
            mlx_held_bytes: 0,
            reading_milliseconds: moment.wired_reading_milliseconds,
        },
    )
}
