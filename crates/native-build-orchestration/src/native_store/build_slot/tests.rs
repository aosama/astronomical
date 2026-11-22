use std::{fs, os::unix::fs::PermissionsExt, sync::mpsc, thread, time::Duration};

use super::{BUILD_SLOT_COUNT, acquire_build_slot};

#[test]
fn should_hand_consecutive_builds_the_same_build_folder() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");

    let first_build_folder = acquire_build_slot(slots_directory.path())
        .expect("a free slot should be taken")
        .cmake_build_directory
        .clone();
    let second_build_folder = acquire_build_slot(slots_directory.path())
        .expect("the released slot should be taken again")
        .cmake_build_directory
        .clone();

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

#[test]
fn should_empty_the_slot_when_the_build_releases_it() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let held_slot = acquire_build_slot(slots_directory.path()).expect("slot 0 should be free");
    let build_folder = held_slot.cmake_build_directory.clone();
    fs::write(build_folder.join("libmlx.a"), b"a large build product").unwrap();

    drop(held_slot);

    assert_eq!(
        fs::read_dir(&build_folder).unwrap().count(),
        0,
        "no build tree may stay on disk"
    );
}

#[test]
fn should_keep_the_slot_locked_in_the_processes_the_build_starts() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let held_slot = acquire_build_slot(slots_directory.path()).expect("slot 0 should be free");

    let descriptor_flags = rustix::io::fcntl_getfd(&held_slot._slot_lock).unwrap();

    assert!(!descriptor_flags.contains(rustix::io::FdFlags::CLOEXEC));
}

#[test]
fn should_skip_a_slot_it_cannot_empty_and_take_the_next() {
    let slots_directory = tempfile::tempdir().expect("the test should create a slots folder");
    let stuck_folder = slots_directory.path().join("slot-0/owned-by-another-user");
    fs::create_dir_all(&stuck_folder).unwrap();
    fs::write(stuck_folder.join("object.o"), b"undeletable").unwrap();
    fs::set_permissions(&stuck_folder, fs::Permissions::from_mode(0o555)).unwrap();

    let taken_slot = acquire_build_slot(slots_directory.path());

    fs::set_permissions(&stuck_folder, fs::Permissions::from_mode(0o755)).unwrap();
    let taken_slot = taken_slot.expect("an unusable slot must not stop the build");
    assert_eq!(
        taken_slot.cmake_build_directory,
        slots_directory.path().join("slot-1")
    );
}

#[test]
fn should_refuse_a_slots_folder_that_is_a_symlink() {
    let store_directory = tempfile::tempdir().expect("the test should create a store folder");
    let elsewhere = tempfile::tempdir().expect("the test should create an unrelated folder");
    let slots_link = store_directory.path().join("build-slots");
    std::os::unix::fs::symlink(elsewhere.path(), &slots_link).unwrap();

    let refusal = acquire_build_slot(&slots_link)
        .err()
        .expect("a symlinked slots folder must be refused");

    assert!(refusal.to_string().contains("symlink"), "{refusal}");
    assert_eq!(
        fs::read_dir(elsewhere.path()).unwrap().count(),
        0,
        "nothing may be written through the link"
    );
}

#[test]
fn should_fail_when_the_slots_folder_is_a_file() {
    let store_directory = tempfile::tempdir().expect("the test should create a store folder");
    let slots_file = store_directory.path().join("build-slots");
    fs::write(&slots_file, b"not a folder").unwrap();

    assert!(acquire_build_slot(&slots_file).is_err());
}
