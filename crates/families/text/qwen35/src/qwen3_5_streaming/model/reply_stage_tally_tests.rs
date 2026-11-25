//! The tally adds what it is told, splits a lookup's time, and keeps the
//! last reply apart from the one being written.

use std::time::Duration;

use serial_test::file_serial;

use super::{DraftRoundStages, ReplyStageCounts, ReplyStageTally};

#[test]
#[file_serial]
fn should_total_a_replys_forwards_lookups_and_rounds() {
    let tally = ReplyStageTally::default();
    tally.begin_reply();
    tally.record_forward(1);
    tally.record_forward(4);
    tally.record_lookup(Duration::from_micros(300), Duration::from_micros(200));
    tally.record_lookup(Duration::from_micros(100), Duration::ZERO);
    tally.record_draft_round(DraftRoundStages {
        draft: Duration::from_millis(4),
        verify: Duration::from_millis(70),
        pick: Duration::from_millis(1),
        keep: Duration::from_micros(150),
    });

    let counts = tally.end_reply().expect("a reply was begun");

    assert_eq!(counts.reply_forward_count, 2);
    assert_eq!(counts.reply_row_count, 5);
    assert_eq!(counts.lookup_count, 2);
    assert_eq!(counts.lookup_nanoseconds, 400_000);
    assert_eq!(counts.lookup_storage_read_nanoseconds, 200_000);
    assert_eq!(
        counts.lookup_bookkeeping_nanoseconds(),
        200_000,
        "what the lookups spent that was not reading"
    );
    assert_eq!(counts.draft_round_count, 1);
    assert_eq!(counts.verify_nanoseconds, 70_000_000);
    assert_eq!(counts.keep_nanoseconds, 150_000);
}

#[test]
#[file_serial]
fn should_keep_the_last_reply_while_the_next_is_written() {
    let tally = ReplyStageTally::default();
    assert_eq!(tally.last_reply(), None, "no reply yet");
    tally.begin_reply();
    tally.record_forward(1);
    let first_reply = tally.end_reply().expect("a reply was begun");

    tally.begin_reply();
    tally.record_forward(1);
    tally.record_forward(1);

    assert_eq!(tally.last_reply(), Some(first_reply));
    assert_eq!(
        tally
            .end_reply()
            .expect("the second reply was begun")
            .reply_forward_count,
        2
    );
}

#[test]
fn should_never_report_negative_bookkeeping_when_a_read_outlasts_its_lookup_clock() {
    // The read is timed inside the fetch and the lookup around it; clock
    // granularity can put the inner reading a few nanoseconds over.
    let counts = ReplyStageCounts {
        lookup_nanoseconds: 100,
        lookup_storage_read_nanoseconds: 104,
        ..ReplyStageCounts::default()
    };

    assert_eq!(counts.lookup_bookkeeping_nanoseconds(), 0);
}

#[test]
#[file_serial]
fn should_close_nothing_for_a_request_that_ended_before_its_reply_began() {
    let tally = ReplyStageTally::default();

    assert_eq!(tally.end_reply(), None);
    assert_eq!(tally.last_reply(), None, "no reply to remember");

    // And a finished reply is not closed a second time.
    tally.begin_reply();
    assert!(tally.end_reply().is_some());
    assert_eq!(tally.end_reply(), None);
}
