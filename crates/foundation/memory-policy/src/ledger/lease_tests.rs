use super::{LeaseBook, LeasePurpose};

#[test]
fn should_hold_nothing_before_any_lease_is_granted() {
    let lease_book = LeaseBook::default();
    assert_eq!(lease_book.held_bytes(), 0);
    assert_eq!(lease_book.held_bytes_for(LeasePurpose::VisionTower), 0);
}

#[test]
fn should_count_every_held_lease_until_it_is_given_back() {
    let mut lease_book = LeaseBook::default();
    let publication_lease = lease_book.grant(LeasePurpose::AstroSsdPromptCachePublication, 300);
    let vision_lease = lease_book.grant(LeasePurpose::VisionTower, 900);
    assert_eq!(lease_book.held_bytes(), 1_200);
    assert_eq!(lease_book.give_back(publication_lease), Some(300));
    assert_eq!(lease_book.held_bytes(), 900);
    assert_eq!(lease_book.give_back(vision_lease), Some(900));
    assert_eq!(lease_book.held_bytes(), 0);
}

#[test]
fn should_total_one_purpose_apart_from_the_others() {
    let mut lease_book = LeaseBook::default();
    lease_book.grant(LeasePurpose::AstroSsdPromptCacheRestore, 100);
    lease_book.grant(LeasePurpose::AstroSsdPromptCacheRestore, 150);
    lease_book.grant(LeasePurpose::DraftModel, 4_000);
    assert_eq!(
        lease_book.held_bytes_for(LeasePurpose::AstroSsdPromptCacheRestore),
        250
    );
    assert_eq!(lease_book.held_bytes_for(LeasePurpose::DraftModel), 4_000);
}

#[test]
fn should_refuse_to_give_the_same_lease_back_twice() {
    let mut lease_book = LeaseBook::default();
    let draft_lease = lease_book.grant(LeasePurpose::DraftModel, 4_000);
    assert_eq!(lease_book.give_back(draft_lease), Some(4_000));
    assert_eq!(lease_book.give_back(draft_lease), None);
    assert_eq!(lease_book.held_bytes(), 0);
}

#[test]
fn should_give_each_lease_its_own_identity_even_at_the_same_size() {
    let mut lease_book = LeaseBook::default();
    let first_lease = lease_book.grant(LeasePurpose::VisionTower, 900);
    let second_lease = lease_book.grant(LeasePurpose::VisionTower, 900);
    assert_ne!(first_lease, second_lease);
    assert_eq!(lease_book.give_back(first_lease), Some(900));
    assert_eq!(lease_book.held_bytes(), 900);
}

#[test]
fn should_saturate_instead_of_wrapping_on_absurd_lease_sizes() {
    let mut lease_book = LeaseBook::default();
    lease_book.grant(LeasePurpose::DraftModel, u64::MAX);
    lease_book.grant(LeasePurpose::DraftModel, 5);
    assert_eq!(lease_book.held_bytes(), u64::MAX);
}

#[test]
fn should_not_count_redo_headroom_as_memory_in_use() {
    let mut lease_book = LeaseBook::default();
    lease_book.grant(LeasePurpose::VisionTower, 900);
    lease_book.grant(LeasePurpose::RedoHeadroom, 500);
    assert_eq!(lease_book.held_bytes(), 1_400);
    assert_eq!(lease_book.allocated_bytes(), 900);
}
