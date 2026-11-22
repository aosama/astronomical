use std::{fs, sync::mpsc, thread, time::Duration};

use super::{BUILD_SLOT_COUNT, acquire_build_slot};

#[test]
fn should_hand_consecutive_builds_the_same_build_folder() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");

    let first_build_folder = acquire_build_slot(slots_directory.path())
        .expect("a free slot should be taken")
        .cmake_build_directory;
    let second_build_folder = acquire_build_slot(slots_directory.path())
        .expect("the released slot should be taken again")
        .cmake_build_directory;

    assert_eq!(first_build_folder, second_build_folder);
    assert_eq!(first_build_folder, slots_directory.path().join("slot-0"));
}

#[test]
fn should_hand_a_concurrent_build_the_next_slot() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");

    let held_slot = acquire_build_slot(slots_directory.path()).expect("slot 0 should be free");
    let concurrent_slot =
        acquire_build_slot(slots_directory.path()).expect("slot 1 should be free");

    assert_eq!(
        held_slot.cmake_build_directory,
        slots_directory.path().join("slot-0")
    );
    assert_eq!(
        concurrent_slot.cmake_build_directory,
        slots_directory.path().join("slot-1")
    );
}

#[test]
fn should_empty_what_an_earlier_build_left_in_the_slot() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let leftover_object = slots_directory.path().join("slot-0/CMakeFiles/stale.o");
    fs::create_dir_all(leftover_object.parent().unwrap()).unwrap();
    fs::write(&leftover_object, b"object from an earlier build").unwrap();

    let held_slot = acquire_build_slot(slots_directory.path()).expect("slot 0 should be free");

    assert!(held_slot.cmake_build_directory.is_dir());
    assert_eq!(
        fs::read_dir(&held_slot.cmake_build_directory)
            .unwrap()
            .count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn should_replace_a_symlink_at_the_slot_path_without_following_it() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let elsewhere = tempfile::tempdir().expect("the test should create an unrelated folder");
    let unrelated_file = elsewhere.path().join("keep-me");
    fs::write(&unrelated_file, b"not part of any build").unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), slots_directory.path().join("slot-0")).unwrap();

    let held_slot = acquire_build_slot(slots_directory.path()).expect("slot 0 should be free");

    assert!(
        !fs::symlink_metadata(&held_slot.cmake_build_directory)
            .unwrap()
            .is_symlink()
    );
    assert!(unrelated_file.is_file(), "the symlink target must survive");
}

#[test]
fn should_wait_for_a_slot_when_every_slot_is_busy() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let mut held_slots = Vec::new();
    for _ in 0..BUILD_SLOT_COUNT {
        held_slots
            .push(acquire_build_slot(slots_directory.path()).expect("each slot should be free"));
    }
    let waiting_slots_directory = slots_directory.path().to_path_buf();
    let (slot_sender, slot_receiver) = mpsc::channel();
    let waiting_builder = thread::spawn(move || {
        let waited_slot =
            acquire_build_slot(&waiting_slots_directory).expect("a slot should free up");
        slot_sender
            .send(waited_slot.cmake_build_directory.clone())
            .unwrap();
    });

    assert!(
        slot_receiver
            .recv_timeout(Duration::from_millis(300))
            .is_err(),
        "no slot is free yet"
    );
    let released_slot = held_slots.remove(2);
    let released_folder = released_slot.cmake_build_directory.clone();
    drop(released_slot);

    let waited_folder = slot_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("the waiting build should get the released slot");
    assert_eq!(waited_folder, released_folder);
    waiting_builder.join().unwrap();
}
