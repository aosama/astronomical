use std::path::Path;

use super::{removal_handoff_path, target_directory_status_line};

#[test]
fn should_name_the_handoff_path_after_the_target_and_the_process_id() {
    let handoff_path = removal_handoff_path(Path::new("/lane-root/target-abc123"), 4242);

    assert_eq!(
        handoff_path,
        Path::new("/lane-root/target-abc123.removing.4242")
    );
}

#[test]
fn should_keep_the_handoff_path_beside_the_target() {
    let target_directory = Path::new("/lane-root/nested/target-abc123");

    let handoff_path = removal_handoff_path(target_directory, 1);

    assert_eq!(handoff_path.parent(), target_directory.parent());
}

#[test]
fn should_build_a_handoff_path_for_a_target_without_a_file_name() {
    // The filesystem root has no file name, so the name part is empty and
    // the handoff path is just the suffix beside the root.
    let handoff_path = removal_handoff_path(Path::new("/"), 7);

    assert_eq!(handoff_path, Path::new("/.removing.7"));
}

#[test]
fn should_put_the_label_before_the_target_directory_in_a_status_line() {
    let status_line =
        target_directory_status_line("cleanup-start", Path::new("/lane-root/target-abc123"));

    assert_eq!(
        status_line,
        "cleanup-start target_directory=/lane-root/target-abc123"
    );
}

#[test]
fn should_keep_the_label_and_spaces_in_the_directory_verbatim_in_a_status_line() {
    let status_line = target_directory_status_line(
        "cleanup-refused-after-handoff",
        Path::new("/lane root/target"),
    );

    assert_eq!(
        status_line,
        "cleanup-refused-after-handoff target_directory=/lane root/target"
    );
}
