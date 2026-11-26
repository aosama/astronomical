use super::SlotReads;

#[test]
fn should_give_up_the_slot_read_the_fewest_times_first() {
    let mut slot_reads = SlotReads::seated(3);
    slot_reads.read(&[0, 1]);
    slot_reads.read(&[0]);

    assert!(slot_reads.giving_up_order(2) < slot_reads.giving_up_order(1));
    assert!(slot_reads.giving_up_order(1) < slot_reads.giving_up_order(0));
}

#[test]
fn should_give_up_the_slot_read_longest_ago_among_equal_counts() {
    let mut slot_reads = SlotReads::seated(2);
    slot_reads.read(&[1]);
    slot_reads.read(&[0]);

    assert!(slot_reads.giving_up_order(1) < slot_reads.giving_up_order(0));
}

#[test]
fn should_rank_a_newly_seated_slot_after_unread_older_ones() {
    let mut slot_reads = SlotReads::seated(2);
    slot_reads.seat(1);

    assert!(slot_reads.giving_up_order(0) < slot_reads.giving_up_order(1));
    assert_eq!(slot_reads.giving_up_order(1).0, 0);
}

#[test]
fn should_start_every_count_again_at_a_reply_but_keep_the_read_order() {
    let mut slot_reads = SlotReads::seated(2);
    slot_reads.read(&[0]);
    slot_reads.read(&[0]);
    slot_reads.read(&[1]);

    slot_reads.start_reply();

    assert_eq!(slot_reads.giving_up_order(0).0, 0);
    assert_eq!(slot_reads.giving_up_order(1).0, 0);
    assert!(slot_reads.giving_up_order(0) < slot_reads.giving_up_order(1));
}

#[test]
fn should_count_grown_slots_as_unread_and_ignore_slots_past_the_table() {
    let mut slot_reads = SlotReads::seated(1);
    slot_reads.read(&[0, 5]);
    slot_reads.grow_to(2);

    assert_eq!(slot_reads.giving_up_order(1), (0, 0));
    assert_eq!(slot_reads.giving_up_order(5), (u64::MAX, u64::MAX));
    assert_eq!(slot_reads.giving_up_order(0).0, 1);
}

#[test]
fn should_seat_over_a_read_slot_with_no_reads() {
    let mut slot_reads = SlotReads::seated(1);
    slot_reads.read(&[0]);
    slot_reads.seat(0);

    assert_eq!(slot_reads.giving_up_order(0).0, 0);
}
