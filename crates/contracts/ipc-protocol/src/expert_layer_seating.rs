//! Per-layer retained counts carried on every residency sample.
//!
//! A full expert-id list on every decode token would dominate the status
//! payload, and the column map only needs how many experts each layer still
//! holds. One hundred twenty-eight covers every sparse model this runner
//! loads; counts past that limit are dropped rather than resized on the
//! hot path. `seated_layer_count` marks the real prefix so unused tail zeros
//! are not published as layers.

use serde::{Deserialize, Serialize};

/// Decoder layers whose retained counts fit in one residency sample.
pub const EXPERT_LAYER_SEATING_LIMIT: usize = 128;

/// How many experts each decoder layer currently keeps in RAM.
///
/// The counts stay in a fixed array so the residency sample can stay `Copy`.
/// Serde in this toolchain only implements arrays through length 32, so the
/// wire form is the used prefix rather than all 128 slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertLayerSeatingCounts {
    pub retained_expert_counts: [u32; EXPERT_LAYER_SEATING_LIMIT],
    /// How many leading entries of `retained_expert_counts` are real layers.
    /// Zero means this sample has no layer map.
    pub seated_layer_count: u32,
}

impl Default for ExpertLayerSeatingCounts {
    fn default() -> Self {
        Self {
            retained_expert_counts: [0; EXPERT_LAYER_SEATING_LIMIT],
            seated_layer_count: 0,
        }
    }
}

#[derive(Deserialize)]
struct ExpertLayerSeatingCountsWire {
    #[serde(default)]
    retained_expert_counts: Vec<u32>,
}

impl Serialize for ExpertLayerSeatingCounts {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut layer_seating = serializer.serialize_struct("ExpertLayerSeatingCounts", 2)?;
        layer_seating.serialize_field("retained_expert_counts", &self.published_counts())?;
        layer_seating.serialize_field("seated_layer_count", &self.seated_layer_count)?;
        layer_seating.end()
    }
}

impl<'de> Deserialize<'de> for ExpertLayerSeatingCounts {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let layer_seating_wire = ExpertLayerSeatingCountsWire::deserialize(deserializer)?;
        Ok(Self::from_retained_counts(
            &layer_seating_wire.retained_expert_counts,
        ))
    }
}

impl ExpertLayerSeatingCounts {
    /// Copies the leading counts, stopping at [`EXPERT_LAYER_SEATING_LIMIT`].
    #[must_use]
    pub fn from_retained_counts(retained_expert_counts: &[u32]) -> Self {
        let mut layer_seating = Self::default();
        let published_layer_count = retained_expert_counts.len().min(EXPERT_LAYER_SEATING_LIMIT);
        let mut layer_index = 0;
        while layer_index < published_layer_count {
            layer_seating.retained_expert_counts[layer_index] = retained_expert_counts[layer_index];
            layer_index += 1;
        }
        layer_seating.seated_layer_count = u32::try_from(published_layer_count).unwrap_or(u32::MAX);
        layer_seating
    }

    /// The real prefix. An empty vector means the sample has no layer map.
    #[must_use]
    pub fn published_counts(&self) -> Vec<u32> {
        let published_layer_count = usize::try_from(self.seated_layer_count).unwrap_or(0);
        let published_layer_count = published_layer_count.min(EXPERT_LAYER_SEATING_LIMIT);
        self.retained_expert_counts[..published_layer_count].to_vec()
    }
}

#[cfg(test)]
mod expert_layer_seating_tests {
    use super::{EXPERT_LAYER_SEATING_LIMIT, ExpertLayerSeatingCounts};
    use crate::protocol_message::WorkerExpertResidencySnapshot;

    #[test]
    fn should_leave_seating_empty_when_no_layers_are_counted() {
        let layer_seating = ExpertLayerSeatingCounts::from_retained_counts(&[]);
        assert_eq!(layer_seating.seated_layer_count, 0);
        assert!(layer_seating.published_counts().is_empty());
        assert_eq!(layer_seating, ExpertLayerSeatingCounts::default());
    }

    #[test]
    fn should_keep_only_the_first_supported_layer_counts() {
        let mut retained_expert_counts = vec![1_u32; EXPERT_LAYER_SEATING_LIMIT + 3];
        retained_expert_counts[0] = 7;
        retained_expert_counts[EXPERT_LAYER_SEATING_LIMIT - 1] = 9;
        let layer_seating = ExpertLayerSeatingCounts::from_retained_counts(&retained_expert_counts);
        assert_eq!(
            layer_seating.seated_layer_count,
            u32::try_from(EXPERT_LAYER_SEATING_LIMIT).unwrap_or(u32::MAX)
        );
        assert_eq!(layer_seating.retained_expert_counts[0], 7);
        assert_eq!(
            layer_seating.retained_expert_counts[EXPERT_LAYER_SEATING_LIMIT - 1],
            9
        );
        assert_eq!(
            layer_seating.published_counts().len(),
            EXPERT_LAYER_SEATING_LIMIT
        );
    }

    #[test]
    fn should_publish_the_exact_prefix_of_retained_counts() {
        let layer_seating = ExpertLayerSeatingCounts::from_retained_counts(&[0, 4, 256]);
        assert_eq!(layer_seating.seated_layer_count, 3);
        assert_eq!(layer_seating.published_counts(), vec![0, 4, 256]);
    }

    #[test]
    fn should_decode_a_residency_snapshot_that_omits_layer_seating() {
        let decoded_residency: WorkerExpertResidencySnapshot = serde_json::from_str(
            r#"{"total_layer_count":1,"resident_expert_count":0,"resident_expert_payload_bytes":0,"complete_expert_payload_byte_count":0,"expert_capacity":0}"#,
        )
        .expect("older residency json should decode");
        assert_eq!(
            decoded_residency.layer_seating,
            ExpertLayerSeatingCounts::default()
        );
        assert!(
            decoded_residency
                .layer_seating
                .published_counts()
                .is_empty()
        );
    }

    #[test]
    fn should_round_trip_the_retained_count_prefix() {
        let layer_seating = ExpertLayerSeatingCounts::from_retained_counts(&[0, 4, 256]);
        let encoded_seating = serde_json::to_string(&layer_seating).expect("seating should encode");
        let decoded_seating: ExpertLayerSeatingCounts =
            serde_json::from_str(&encoded_seating).expect("seating should decode");
        assert_eq!(decoded_seating, layer_seating);
        assert_eq!(
            encoded_seating,
            r#"{"retained_expert_counts":[0,4,256],"seated_layer_count":3}"#
        );
    }
}
