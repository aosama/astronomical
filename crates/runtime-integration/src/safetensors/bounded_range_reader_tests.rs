//! Span planning for the bounded multi-range reader: how one virtual span
//! MLX asks for is cut into per-interval source reads.

use super::{IntervalPiece, plan_interval_pieces};
use crate::safetensors::lazy_map::BoundedReadInterval;

/// Three intervals tiling a 60-byte virtual payload from scattered source
/// offsets, the shape a routed page of three experts produces.
fn three_scattered_intervals() -> Vec<BoundedReadInterval> {
    vec![
        BoundedReadInterval {
            virtual_payload_offset: 0,
            source_file_offset: 1_000,
            source_byte_count: 20,
        },
        BoundedReadInterval {
            virtual_payload_offset: 20,
            source_file_offset: 5_000,
            source_byte_count: 20,
        },
        BoundedReadInterval {
            virtual_payload_offset: 40,
            source_file_offset: 9_000,
            source_byte_count: 20,
        },
    ]
}

fn piece_triples(interval_pieces: &[IntervalPiece]) -> Vec<(usize, usize, u64)> {
    interval_pieces
        .iter()
        .map(|interval_piece| {
            (
                interval_piece.destination_offset,
                interval_piece.byte_count,
                interval_piece.source_file_offset,
            )
        })
        .collect()
}

#[test]
fn should_plan_one_piece_per_interval_for_a_span_covering_every_interval() {
    let interval_pieces =
        plan_interval_pieces(&three_scattered_intervals(), 0, 60, 0).expect("the span is covered");
    assert_eq!(
        piece_triples(&interval_pieces),
        vec![(0, 20, 1_000), (20, 20, 5_000), (40, 20, 9_000)]
    );
}

#[test]
fn should_plan_a_single_piece_for_a_span_inside_one_interval() {
    let interval_pieces =
        plan_interval_pieces(&three_scattered_intervals(), 0, 6, 22).expect("the span is covered");
    assert_eq!(piece_triples(&interval_pieces), vec![(0, 6, 5_002)]);
}

#[test]
fn should_trim_the_first_and_last_pieces_for_a_span_starting_and_ending_mid_interval() {
    let interval_pieces =
        plan_interval_pieces(&three_scattered_intervals(), 0, 30, 15).expect("the span is covered");
    assert_eq!(
        piece_triples(&interval_pieces),
        vec![(0, 5, 1_015), (5, 20, 5_000), (25, 5, 9_000)]
    );
}

#[test]
fn should_offset_every_piece_by_the_destination_start() {
    let interval_pieces = plan_interval_pieces(&three_scattered_intervals(), 100, 40, 0)
        .expect("the span is covered");
    assert_eq!(
        piece_triples(&interval_pieces),
        vec![(100, 20, 1_000), (120, 20, 5_000)]
    );
}

#[test]
fn should_plan_a_one_byte_span_at_the_last_payload_byte() {
    let interval_pieces =
        plan_interval_pieces(&three_scattered_intervals(), 0, 1, 59).expect("the span is covered");
    assert_eq!(piece_triples(&interval_pieces), vec![(0, 1, 9_019)]);
}

#[test]
fn should_refuse_a_span_that_starts_in_a_gap_between_intervals() {
    let gapped_intervals = vec![
        BoundedReadInterval {
            virtual_payload_offset: 0,
            source_file_offset: 0,
            source_byte_count: 10,
        },
        BoundedReadInterval {
            virtual_payload_offset: 20,
            source_file_offset: 100,
            source_byte_count: 10,
        },
    ];
    assert!(plan_interval_pieces(&gapped_intervals, 0, 5, 12).is_none());
}

#[test]
fn should_refuse_a_span_that_runs_past_the_last_interval() {
    assert!(plan_interval_pieces(&three_scattered_intervals(), 0, 10, 55).is_none());
}

#[test]
fn should_end_a_span_exactly_on_an_interval_boundary() {
    let interval_pieces =
        plan_interval_pieces(&three_scattered_intervals(), 0, 20, 20).expect("the span is covered");
    assert_eq!(piece_triples(&interval_pieces), vec![(0, 20, 5_000)]);
}
