use crate::install_verification_tools::{
    cargo_about_archive_name, cargo_about_release_url, curl_download_arguments,
    expected_version_line, host_rust_target, published_digest_from_sidecar, sccache_archive_name,
    sccache_release_url,
};

fn curl_option_value(option_name: &str) -> Option<&'static str> {
    let download_arguments = curl_download_arguments();
    let option_position = download_arguments
        .iter()
        .position(|argument| *argument == option_name)?;
    download_arguments.get(option_position + 1).copied()
}

#[test]
fn should_back_off_exponentially_instead_of_retrying_on_a_fixed_delay() {
    assert!(
        !curl_download_arguments().contains(&"--retry-delay"),
        "a fixed --retry-delay disables curl's exponential backoff"
    );
    let retry_attempts: u32 = curl_option_value("--retry")
        .expect("downloads must retry")
        .parse()
        .expect("--retry takes a count");
    assert!(
        retry_attempts >= 6,
        "six retries span about a minute of backoff, enough to outlast a release-edge 504"
    );
}

/// The 1+2+4+8+16+32 s backoff of six retries.
const SIX_RETRY_BACKOFF_SECONDS: u32 = 63;

fn curl_seconds_option(option_name: &str) -> u32 {
    curl_option_value(option_name)
        .unwrap_or_else(|| panic!("{option_name} must be set"))
        .parse()
        .unwrap_or_else(|_| panic!("{option_name} takes seconds"))
}

#[test]
fn should_retry_receive_errors_but_bound_the_whole_window() {
    assert!(curl_download_arguments().contains(&"--retry-all-errors"));
    let retry_window_seconds = curl_seconds_option("--retry-max-time");
    assert!(
        retry_window_seconds >= SIX_RETRY_BACKOFF_SECONDS,
        "a window shorter than the backoff would silently drop retries"
    );
    assert!(
        retry_window_seconds <= 180,
        "the window must still fail a missing archive quickly"
    );
}

#[test]
fn should_bound_each_attempt_so_a_stalled_transfer_cannot_outlive_the_window() {
    let connect_timeout_seconds = curl_seconds_option("--connect-timeout");
    let attempt_timeout_seconds = curl_seconds_option("--max-time");
    assert!(connect_timeout_seconds > 0 && connect_timeout_seconds < attempt_timeout_seconds);
    assert!(
        attempt_timeout_seconds <= curl_seconds_option("--retry-max-time"),
        "one attempt must not run longer than the whole retry window"
    );
}

#[test]
fn should_keep_https_only_and_fail_on_http_errors() {
    let download_arguments = curl_download_arguments();
    assert!(download_arguments.contains(&"--fail"));
    assert_eq!(curl_option_value("--proto"), Some("=https"));
    assert!(download_arguments.contains(&"--tlsv1.2"));
}

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
