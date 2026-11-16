//! Whether one more model fits on the disk beside everything already listed.
//!
//! Each download still checks the disk for itself when its turn comes. This
//! check runs earlier, when the person presses Get: a list that cannot fit
//! would otherwise be accepted now and fail hours later, after the person
//! has walked away.

/// The bytes the whole list needs and the bytes the disk has, when the list
/// would not fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueDiskShortfall {
    pub required_bytes: u64,
    pub available_bytes: u64,
}

/// Admits a model when the bytes still to fetch for the current download,
/// every waiting model and the new one, plus a one-percent margin, fit in the
/// free space. The margin matches the one each download's own check keeps
/// for staging, so a list admitted here does not fail that later check on
/// rounding alone.
pub fn admit_into_queue(
    available_bytes: u64,
    current_download_remaining_bytes: u64,
    waiting_model_sizes_bytes: &[u64],
    new_model_size_bytes: u64,
) -> Result<(), QueueDiskShortfall> {
    let listed_bytes = waiting_model_sizes_bytes
        .iter()
        .fold(
            current_download_remaining_bytes,
            |total_bytes, waiting_bytes| total_bytes.saturating_add(*waiting_bytes),
        )
        .saturating_add(new_model_size_bytes);
    let required_bytes = listed_bytes.saturating_add(listed_bytes.div_ceil(100).max(1));
    if required_bytes > available_bytes {
        return Err(QueueDiskShortfall {
            required_bytes,
            available_bytes,
        });
    }
    Ok(())
}
