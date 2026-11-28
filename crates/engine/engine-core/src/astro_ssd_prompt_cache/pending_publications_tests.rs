use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use super::AstroSsdPromptCachePendingPublications;

/// Long enough for a waiting thread to have blocked, short enough to keep
/// the suite quick; the assertions only need the order of events.
const SETTLE_DURATION: Duration = Duration::from_millis(50);

#[test]
fn should_count_an_admitted_block_and_release_it_on_completion() {
    let pending_publications = AstroSsdPromptCachePendingPublications::new();
    pending_publications.admit(100, 1_000);
    assert_eq!(pending_publications.queued_block_count(), 1);
    assert_eq!(pending_publications.queued_payload_bytes(), 100);
    pending_publications.complete(100);
    assert_eq!(pending_publications.queued_block_count(), 0);
    assert_eq!(pending_publications.queued_payload_bytes(), 0);
}

#[test]
fn should_admit_a_block_larger_than_the_bound_into_an_empty_queue() {
    let pending_publications = AstroSsdPromptCachePendingPublications::new();
    pending_publications.admit(5_000, 1_000);
    assert_eq!(pending_publications.queued_payload_bytes(), 5_000);
}

#[test]
fn should_admit_a_block_that_exactly_fills_the_bound() {
    let pending_publications = AstroSsdPromptCachePendingPublications::new();
    pending_publications.admit(600, 1_000);
    pending_publications.admit(400, 1_000);
    assert_eq!(pending_publications.queued_block_count(), 2);
}

#[test]
fn should_hold_a_block_that_would_pass_the_bound_until_room_is_made() {
    let pending_publications = Arc::new(AstroSsdPromptCachePendingPublications::new());
    pending_publications.admit(800, 1_000);
    let is_second_block_admitted = Arc::new(AtomicBool::new(false));
    let admitting_thread = {
        let pending_publications = Arc::clone(&pending_publications);
        let is_second_block_admitted = Arc::clone(&is_second_block_admitted);
        thread::spawn(move || {
            pending_publications.admit(800, 1_000);
            is_second_block_admitted.store(true, Ordering::SeqCst);
        })
    };
    thread::sleep(SETTLE_DURATION);
    assert!(!is_second_block_admitted.load(Ordering::SeqCst));
    pending_publications.complete(800);
    admitting_thread
        .join()
        .expect("the admitting thread finishes");
    assert!(is_second_block_admitted.load(Ordering::SeqCst));
    assert_eq!(pending_publications.queued_payload_bytes(), 800);
}

#[test]
fn should_return_at_once_when_nothing_is_queued() {
    let pending_publications = AstroSsdPromptCachePendingPublications::new();
    pending_publications.wait_until_drained();
    assert_eq!(pending_publications.queued_block_count(), 0);
}

#[test]
fn should_hold_a_reader_until_every_queued_block_is_finished() {
    let pending_publications = Arc::new(AstroSsdPromptCachePendingPublications::new());
    pending_publications.admit(10, 1_000);
    pending_publications.admit(20, 1_000);
    let has_reader_passed = Arc::new(AtomicBool::new(false));
    let reading_thread = {
        let pending_publications = Arc::clone(&pending_publications);
        let has_reader_passed = Arc::clone(&has_reader_passed);
        thread::spawn(move || {
            pending_publications.wait_until_drained();
            has_reader_passed.store(true, Ordering::SeqCst);
        })
    };
    pending_publications.complete(10);
    thread::sleep(SETTLE_DURATION);
    assert!(!has_reader_passed.load(Ordering::SeqCst));
    pending_publications.complete(20);
    reading_thread.join().expect("the reading thread finishes");
    assert!(has_reader_passed.load(Ordering::SeqCst));
}

#[test]
fn should_not_underflow_on_a_completion_without_an_admission() {
    let pending_publications = AstroSsdPromptCachePendingPublications::new();
    pending_publications.complete(10);
    assert_eq!(pending_publications.queued_block_count(), 0);
    assert_eq!(pending_publications.queued_payload_bytes(), 0);
}
