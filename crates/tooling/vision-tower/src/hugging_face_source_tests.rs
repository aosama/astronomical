use super::*;

#[test]
fn should_read_a_listing_and_pin_its_commit() {
    let listing = listing_from_reply(
        "example-org/model",
        br#"{"sha": "abc123", "pipeline_tag": "text-generation",
             "cardData": {"license": "mit"},
             "siblings": [{"rfilename": "config.json", "size": 12},
                          {"rfilename": "optiq/tower.safetensors"}]}"#,
    )
    .unwrap();
    assert_eq!(listing.pinned_revision.as_deref(), Some("abc123"));
    assert_eq!(listing.license.as_deref(), Some("mit"));
    assert_eq!(listing.files.len(), 2);
    assert_eq!(listing.files[0].byte_count, Some(12));
    assert_eq!(listing.files[1].byte_count, None);
}

#[test]
fn should_join_a_list_of_licenses_and_tolerate_none() {
    let listed = listing_from_reply(
        "example-org/model",
        br#"{"sha": "a", "siblings": [], "cardData": {"license": ["mit", "apache-2.0"]}}"#,
    )
    .unwrap();
    assert_eq!(listed.license.as_deref(), Some("mit, apache-2.0"));
    let absent = listing_from_reply("example-org/model", br#"{"sha": "a", "siblings": []}"#);
    assert_eq!(absent.unwrap().license, None);
}

#[test]
fn should_refuse_a_listing_that_is_not_the_expected_document() {
    assert!(matches!(
        listing_from_reply("example-org/model", b"{\"error\": \"nope\"}"),
        Err(VisionTowerError::RepositoryListingMalformed { .. })
    ));
}

#[test]
fn should_drop_listed_names_that_climb_out_of_the_repository() {
    let listing = listing_from_reply(
        "example-org/model",
        br#"{"sha": "a", "siblings": [{"rfilename": "../secret"},
             {"rfilename": "a//b"}, {"rfilename": "./c"}, {"rfilename": "ok.json"}]}"#,
    )
    .unwrap();
    assert_eq!(listing.files.len(), 1);
    assert_eq!(listing.files[0].relative_path, "ok.json");
}

#[test]
fn should_build_pinned_urls_that_encode_file_names() {
    let url = resolve_url("example-org/model", "abc123", "optiq/a b#c.safetensors").unwrap();
    assert_eq!(
        url.as_str(),
        "https://huggingface.co/example-org/model/resolve/abc123/optiq/a%20b%23c.safetensors"
    );
    let listing = listing_url("example-org/model", "main").unwrap();
    assert_eq!(
        listing.as_str(),
        "https://huggingface.co/api/models/example-org/model/revision/main?blobs=true"
    );
}

#[test]
fn should_accept_only_the_exact_range_requested() {
    assert!(content_range_matches("bytes 8-15/5000", 8, 15));
    assert!(content_range_matches("bytes 8-15/*", 8, 15));
    assert!(!content_range_matches("bytes 0-15/5000", 8, 15));
    assert!(!content_range_matches("bytes 8-16/5000", 8, 15));
    assert!(!content_range_matches("items 8-15/5000", 8, 15));
    assert!(!content_range_matches("bytes 8-15", 8, 15));
    assert!(!content_range_matches("", 8, 15));
}
