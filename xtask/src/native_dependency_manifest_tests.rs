use astronomical_test_harness::offline_test;

use crate::native_dependency_manifest::{parse_manifest_line, parse_manifest_text};

const VALID_SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn valid_manifest_text() -> String {
    [
        "mlx-0.1.tar.gz|https://example.com/mlx-0.1.tar.gz|".to_owned() + VALID_SHA256 + "|MLX 0.1",
        "mlx-c-0.1.tar.gz|https://example.com/mlx-c-0.1.tar.gz|".to_owned()
            + VALID_SHA256
            + "|MLX C 0.1",
        "metal.tar.gz|https://example.com/metal.tar.gz|".to_owned() + VALID_SHA256 + "|metal-cpp",
        "json.tar.gz|https://example.com/json.tar.gz|".to_owned() + VALID_SHA256 + "|json",
        "fmt.tar.gz|https://example.com/fmt.tar.gz|".to_owned() + VALID_SHA256 + "|fmt",
    ]
    .join("\n")
        + "\n"
}

offline_test!(manifest_line_parses_four_fields, {
    let manifest_entry = parse_manifest_line(
        &("mlx.tar.gz|https://example.com/a.tar.gz|".to_owned() + VALID_SHA256 + "|MLX 0.1"),
    )
    .unwrap();
    assert_eq!(manifest_entry.archive_file_name, "mlx.tar.gz");
    assert_eq!(manifest_entry.archive_url, "https://example.com/a.tar.gz");
    assert_eq!(manifest_entry.archive_sha256, VALID_SHA256);
    assert_eq!(manifest_entry.dependency_description, "MLX 0.1");
});

offline_test!(manifest_line_rejects_wrong_field_count, {
    assert!(parse_manifest_line("only-one-field").is_err());
    assert!(parse_manifest_line("a|b|c").is_err());
    assert!(parse_manifest_line("a|b|c|d|e").is_err());
});

offline_test!(manifest_field_rejects_empty_and_line_breaks, {
    assert!(
        parse_manifest_line(&("|https://example.com/a|".to_owned() + VALID_SHA256 + "|d")).is_err()
    );
    let newline_sha256 = "abc\n".to_owned() + VALID_SHA256;
    assert!(
        parse_manifest_line(
            &("a.tar.gz|https://example.com/a.tar.gz|".to_owned() + &newline_sha256 + "|d")
        )
        .is_err()
    );
    let carriage_sha256 = "abc\r".to_owned() + VALID_SHA256;
    assert!(
        parse_manifest_line(
            &("a.tar.gz|https://example.com/a.tar.gz|".to_owned() + &carriage_sha256 + "|d")
        )
        .is_err()
    );
});

offline_test!(archive_file_name_charset_is_enforced, {
    let make_line = |file_name: &str| {
        file_name.to_owned() + "|https://example.com/a.tar.gz|" + VALID_SHA256 + "|d"
    };
    assert!(parse_manifest_line(&make_line("mlx-0.1_v2.tar.gz")).is_ok());
    assert!(parse_manifest_line(&make_line("")).is_err());
    assert!(parse_manifest_line(&make_line("../escape.tar.gz")).is_err());
    assert!(parse_manifest_line(&make_line("space name.tar.gz")).is_err());
    assert!(parse_manifest_line(&make_line("pipe|name.tar.gz")).is_err());
    assert!(parse_manifest_line(&make_line("unicode-\u{00e9}.tar.gz")).is_err());
});

offline_test!(archive_url_requires_https, {
    let make_line = |url: &str| "a.tar.gz|".to_owned() + url + "|" + VALID_SHA256 + "|d";
    assert!(parse_manifest_line(&make_line("https://example.com/a.tar.gz")).is_ok());
    assert!(parse_manifest_line(&make_line("http://example.com/a.tar.gz")).is_err());
    assert!(parse_manifest_line(&make_line("ftp://example.com/a.tar.gz")).is_err());
    assert!(parse_manifest_line(&make_line("example.com/a.tar.gz")).is_err());
});

offline_test!(archive_sha256_requires_lowercase_64_hex, {
    let make_line =
        |digest: &str| "a.tar.gz|https://example.com/a.tar.gz|".to_owned() + digest + "|d";
    assert!(parse_manifest_line(&make_line(VALID_SHA256)).is_ok());
    assert!(parse_manifest_line(&make_line(&VALID_SHA256.to_uppercase())).is_err());
    assert!(parse_manifest_line(&make_line(&("g".to_owned() + &VALID_SHA256[1..]))).is_err());
    assert!(parse_manifest_line(&make_line(&VALID_SHA256[..63])).is_err());
    assert!(parse_manifest_line(&make_line(&(VALID_SHA256.to_owned() + "a"))).is_err());
});

offline_test!(manifest_requires_exactly_five_entries, {
    assert!(parse_manifest_text(&valid_manifest_text()).is_ok());
    let four_entries = valid_manifest_text()
        .lines()
        .take(4)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(parse_manifest_text(&four_entries).is_err());
    let six_entries = valid_manifest_text() + valid_manifest_text().lines().next().unwrap();
    assert!(parse_manifest_text(&six_entries).is_err());
});

offline_test!(empty_manifest_line_is_rejected, {
    let five_with_blank_line =
        valid_manifest_text().replace("mlx-c-0.1.tar.gz", "\nmlx-c-0.1.tar.gz");
    assert!(parse_manifest_text(&five_with_blank_line).is_err());
});
