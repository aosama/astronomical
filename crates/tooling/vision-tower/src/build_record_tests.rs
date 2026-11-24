use super::*;

#[test]
fn should_hash_a_file_to_its_known_digest() {
    let directory = tempfile::tempdir().unwrap();
    let file_path = directory.path().join("known.txt");
    std::fs::write(&file_path, "abc").unwrap();
    let (digest, byte_count) = hash_file(&file_path).unwrap();
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(byte_count, 3);
}

#[test]
fn should_hash_an_empty_file_and_fail_on_a_missing_one() {
    let directory = tempfile::tempdir().unwrap();
    let file_path = directory.path().join("empty.txt");
    std::fs::write(&file_path, "").unwrap();
    let (digest, byte_count) = hash_file(&file_path).unwrap();
    assert_eq!(
        digest,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(byte_count, 0);
    assert!(hash_file(&directory.path().join("absent.txt")).is_err());
}
