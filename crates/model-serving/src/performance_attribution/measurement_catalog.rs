use std::time::Duration;

/// Bounded aggregate for every occurrence of one operation in one report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PerformanceOperationMeasurement {
    pub(super) occurrence_count: u64,
    pub(super) total_elapsed_nanoseconds: u64,
    pub(super) minimum_elapsed_nanoseconds: u64,
    pub(super) maximum_elapsed_nanoseconds: u64,
    pub(super) first_started_offset_nanoseconds: u64,
    pub(super) last_ended_offset_nanoseconds: u64,
}

impl PerformanceOperationMeasurement {
    pub(super) const EMPTY: Self = Self {
        occurrence_count: 0,
        total_elapsed_nanoseconds: 0,
        minimum_elapsed_nanoseconds: u64::MAX,
        maximum_elapsed_nanoseconds: 0,
        first_started_offset_nanoseconds: 0,
        last_ended_offset_nanoseconds: 0,
    };

    #[must_use]
    pub const fn occurrence_count(self) -> u64 {
        self.occurrence_count
    }

    #[must_use]
    pub const fn total_elapsed_nanoseconds(self) -> u64 {
        self.total_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn minimum_elapsed_nanoseconds(self) -> u64 {
        self.minimum_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn maximum_elapsed_nanoseconds(self) -> u64 {
        self.maximum_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn first_started_offset_nanoseconds(self) -> u64 {
        self.first_started_offset_nanoseconds
    }

    #[must_use]
    pub const fn last_ended_offset_nanoseconds(self) -> u64 {
        self.last_ended_offset_nanoseconds
    }

    pub(super) fn record(&mut self, started_offset: Duration, ended_offset: Duration) {
        let started_offset_nanoseconds = duration_nanoseconds_saturating(started_offset);
        let ended_offset_nanoseconds = duration_nanoseconds_saturating(ended_offset);
        let elapsed_nanoseconds =
            ended_offset_nanoseconds.saturating_sub(started_offset_nanoseconds);
        if self.occurrence_count == 0 {
            self.first_started_offset_nanoseconds = started_offset_nanoseconds;
        }
        self.occurrence_count = self.occurrence_count.saturating_add(1);
        self.total_elapsed_nanoseconds = self
            .total_elapsed_nanoseconds
            .saturating_add(elapsed_nanoseconds);
        self.minimum_elapsed_nanoseconds =
            self.minimum_elapsed_nanoseconds.min(elapsed_nanoseconds);
        self.maximum_elapsed_nanoseconds =
            self.maximum_elapsed_nanoseconds.max(elapsed_nanoseconds);
        self.last_ended_offset_nanoseconds = ended_offset_nanoseconds;
    }
}

fn duration_nanoseconds_saturating(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
