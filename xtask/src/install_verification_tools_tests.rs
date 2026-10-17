use crate::install_verification_tools::{
    cargo_about_archive_name, cargo_about_release_url, expected_version_line, host_rust_target,
    published_digest_from_sidecar, read_dotted_version_pin, sccache_archive_name,
    sccache_release_url,
};

#[test]
fn host_mapping_covers_both_runner_platforms_and_fails_closed() {
    assert_eq!(
        host_rust_target("Darwin", "arm64"),
        Ok("aarch64-apple-darwin")
    );
    assert_eq!(
        host_rust_target("Darwin", "x86_64"),
        Ok("x86_64-apple-darwin")
    );
    assert_eq!(
        host_rust_target("Linux", "aarch64"),
        Ok("aarch64-unknown-linux-musl")
    );
    assert_eq!(
        host_rust_target("Linux", "x86_64"),
        Ok("x86_64-unknown-linux-musl")
    );
    assert!(host_rust_target("Windows", "x86_64").is_err());
    assert!(host_rust_target("SunOS", "sparc").is_err());
}

#[test]
fn pin_reading_accepts_only_dotted_versions() {
    assert_eq!(read_dotted_version_pin("0.9.2\n"), Ok("0.9.2".to_owned()));
    assert_eq!(read_dotted_version_pin("1.2"), Ok("1.2".to_owned()));
    assert!(read_dotted_version_pin("").is_err());
    assert!(read_dotted_version_pin("latest\n").is_err());
    assert!(read_dotted_version_pin("0.9.2-beta").is_err());
}

#[test]
fn archive_names_and_release_urls_match_the_pinned_publishers() {
    assert_eq!(
        cargo_about_archive_name("0.9.2", "aarch64-apple-darwin"),
        "cargo-about-0.9.2-aarch64-apple-darwin.tar.gz"
    );
    assert_eq!(
        sccache_archive_name("0.9.1", "aarch64-apple-darwin"),
        "sccache-v0.9.1-aarch64-apple-darwin.tar.gz"
    );
    let cargo_about_url = cargo_about_release_url(
        "0.9.2",
        &cargo_about_archive_name("0.9.2", "aarch64-apple-darwin"),
    );
    assert_eq!(
        cargo_about_url,
        "https://github.com/EmbarkStudios/cargo-about/releases/download/0.9.2/cargo-about-0.9.2-aarch64-apple-darwin.tar.gz"
    );
    let sccache_url = sccache_release_url(
        "0.9.1",
        &sccache_archive_name("0.9.1", "aarch64-apple-darwin"),
    );
    assert_eq!(
        sccache_url,
        "https://github.com/mozilla/sccache/releases/download/v0.9.1/sccache-v0.9.1-aarch64-apple-darwin.tar.gz"
    );
}

#[test]
fn version_lines_use_the_publisher_reported_spelling() {
    assert_eq!(
        expected_version_line("cargo-about", "0.9.2"),
        "cargo-about 0.9.2"
    );
    assert_eq!(expected_version_line("sccache", "0.9.1"), "sccache 0.9.1");
}

#[test]
fn sidecar_parsing_finds_the_digest_hex_token() {
    assert_eq!(
        published_digest_from_sidecar(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  cargo-about.tar.gz\n"
        ),
        Ok("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned())
    );
    // Uppercase hex is normalised to lowercase for comparison.
    assert_eq!(
        published_digest_from_sidecar(
            "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789"
        )
        .unwrap()
        .len(),
        64
    );
    assert!(published_digest_from_sidecar("no digest here").is_err());
    assert!(published_digest_from_sidecar("").is_err());
}
