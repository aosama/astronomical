//! Hermetic contracts for bounded immutable Hugging Face manifest discovery.

use std::{future::Future, time::Duration};

use astronomical_supervisor::{
    DownloadCatalog, DownloadFileDigest, DownloadJob, HubHttpRequest, HubHttpResponse,
    HubHttpResponseError, HubPayloadRequest, HubPayloadTransport, HubTransport, HuggingFaceHub,
    HuggingFaceHubError, HuggingFaceHubLimits, ReqwestHubTransport,
};

mod support;

use support::{
    ScriptedExchange, announce, hub_with_tree, metadata_url, scripted_transport, tree_url,
    valid_metadata_exchange,
};

#[tokio::test]
async fn should_build_an_exact_manifest_from_only_release_selected_payload_paths() {
    announce("release-selected executable payload manifest");
    let transport = scripted_transport([
        ScriptedExchange::json(
            metadata_url(),
            200,
            serde_json::json!({
                "id": REPOSITORY_ID,
                "sha": REVISION,
                "private": false,
                "gated": false
            }),
        ),
        ScriptedExchange::json(
            tree_url(),
            200,
            serde_json::json!([
                {"type":"file","size":2,"path":"model_index.json","oid":GIT_SHA1},
                {"type":"file","size":7,"path":"transformer/model.safetensors","oid":GIT_SHA1},
                {"type":"file","size":99,"path":"alternate-model.safetensors","oid":GIT_SHA1}
            ]),
        ),
    ]);
    let catalog = DownloadCatalog::parse_json(&format!(
        "{{\"schema_version\":1,\"entries\":[{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"display_name\":\"Example\",\"family\":\"flux2_klein\",\"approximate_size_bytes\":9,\"public\":true,\"included_paths\":[\"model_index.json\",\"transformer/\"]}}]}}"
    ))
    .expect("the selected fixture catalog should parse");
    let hub = HuggingFaceHub::new(transport);

    let manifest = bounded(hub.fetch_selected_manifest(
        REPOSITORY_ID,
        REVISION,
        catalog.entries()[0].download_path_selection(),
    ))
    .await
    .expect("only selected executable files should form the manifest");

    assert_eq!(manifest.total_bytes(), 9);
    assert_eq!(
        manifest
            .files()
            .iter()
            .map(|file| file.relative_path())
            .collect::<Vec<_>>(),
        ["model_index.json", "transformer/model.safetensors"]
    );
}

const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const GIT_SHA1: &str = "1111111111111111111111111111111111111111";
const LFS_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const XET_HASH: &str = "3333333333333333333333333333333333333333333333333333333333333333";

#[tokio::test]
async fn should_fetch_exact_metadata_before_following_a_recursive_paginated_tree() {
    announce("immutable metadata and recursive tree journey");
    let second_page_url = format!(
        "https://huggingface.co/api/models/{REPOSITORY_ID}/tree/{REVISION}?recursive=true&cursor=next"
    );
    let transport = scripted_transport([
        ScriptedExchange::json(
            metadata_url(),
            200,
            serde_json::json!({
                "id": REPOSITORY_ID,
                "sha": REVISION,
                "private": false,
                "gated": false,
                "unknownMetadata": {"isIgnored": true}
            }),
        ),
        ScriptedExchange::json_with_headers(
            tree_url(),
            200,
            [("link", format!("<{second_page_url}>; rel=\"next\""))],
            serde_json::json!([
                {
                    "type": "directory",
                    "size": 0,
                    "path": "weights",
                    "oid": GIT_SHA1,
                    "unknownTreeField": true
                },
                {
                    "type": "file",
                    "size": 2,
                    "path": "config.json",
                    "oid": GIT_SHA1
                }
            ]),
        ),
        ScriptedExchange::json(
            second_page_url,
            200,
            serde_json::json!([
                {
                    "type": "file",
                    "size": 7,
                    "path": "weights/model.safetensors",
                    "oid": GIT_SHA1,
                    "lfs": {"oid": LFS_SHA256, "size": 7, "pointerSize": 128},
                    "xetHash": XET_HASH
                }
            ]),
        ),
    ]);
    let hub = HuggingFaceHub::new(transport.clone());

    let manifest = bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
        .await
        .expect("a complete public immutable manifest should be accepted");

    assert_eq!(manifest.repository_id(), REPOSITORY_ID);
    assert_eq!(manifest.revision(), REVISION);
    assert_eq!(manifest.total_bytes(), 9);
    assert_eq!(manifest.files().len(), 2);
    assert_eq!(
        manifest.files()[0].digest(),
        &DownloadFileDigest::GitBlobSha1(GIT_SHA1.to_owned())
    );
    assert_eq!(
        manifest.files()[1].digest(),
        &DownloadFileDigest::Sha256(LFS_SHA256.to_owned())
    );
    assert_eq!(manifest.files()[1].xet_hash(), Some(XET_HASH));
    let durable_job = DownloadJob::from_manifest(&manifest, 1_000)
        .expect("the exact manifest should become durable transfer state");
    assert_eq!(durable_job.remaining_bytes(), 9);
    assert_eq!(
        durable_job.files()[0].expected_digest(),
        &DownloadFileDigest::GitBlobSha1(GIT_SHA1.to_owned())
    );
    assert_eq!(
        durable_job.files()[1].expected_digest(),
        &DownloadFileDigest::Sha256(LFS_SHA256.to_owned())
    );
    assert_eq!(transport.remaining_exchange_count(), 0);
}

#[tokio::test]
async fn should_map_unauthorized_or_forbidden_metadata_to_download_gated() {
    announce("gated metadata status mapping");
    for gated_status in [401, 403] {
        let hub = HuggingFaceHub::new(scripted_transport([ScriptedExchange::empty(
            metadata_url(),
            gated_status,
        )]));

        let manifest_error = bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
            .await
            .expect_err("authentication-requiring repositories must fail as gated");

        assert!(matches!(manifest_error, HuggingFaceHubError::DownloadGated));
    }

    for gated_metadata in [
        serde_json::json!({"id": REPOSITORY_ID, "sha": REVISION, "private": true, "gated": false}),
        serde_json::json!({"id": REPOSITORY_ID, "sha": REVISION, "private": false, "gated": "manual"}),
    ] {
        let hub = HuggingFaceHub::new(scripted_transport([ScriptedExchange::json(
            metadata_url(),
            200,
            gated_metadata,
        )]));
        assert!(matches!(
            bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
            Err(HuggingFaceHubError::DownloadGated)
        ));
    }
}

#[tokio::test]
async fn should_preserve_non_gated_http_status_as_typed_failure() {
    for unexpected_status in [404, 429, 500] {
        let hub = HuggingFaceHub::new(scripted_transport([ScriptedExchange::empty(
            metadata_url(),
            unexpected_status,
        )]));

        assert!(matches!(
            bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
            Err(HuggingFaceHubError::UnexpectedStatus { status }) if status == unexpected_status
        ));
    }
}

#[tokio::test]
async fn should_fail_closed_when_metadata_does_not_prove_the_requested_public_revision() {
    announce("metadata revision and visibility validation");
    let invalid_metadata_documents = [
        serde_json::json!({"sha": GIT_SHA1, "private": false, "gated": false}),
        serde_json::json!({"sha": REVISION, "private": true, "gated": false}),
        serde_json::json!({"sha": REVISION, "private": false, "gated": "manual"}),
        serde_json::json!({"sha": REVISION, "private": "false", "gated": false}),
        serde_json::json!({"sha": REVISION, "private": false}),
        serde_json::json!({"sha": REVISION, "gated": false}),
        serde_json::json!({"sha": REVISION, "private": false, "gated": false}),
        serde_json::json!({"private": false, "gated": false}),
    ];

    for invalid_metadata_document in invalid_metadata_documents {
        let transport = scripted_transport([ScriptedExchange::json(
            metadata_url(),
            200,
            invalid_metadata_document,
        )]);
        let hub = HuggingFaceHub::new(transport.clone());

        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
            .await
            .expect_err("unproven metadata must stop before tree discovery");
        assert_eq!(transport.remaining_exchange_count(), 0);
    }
}

#[tokio::test]
async fn should_construct_production_transport_without_network_and_reject_untrusted_origins() {
    announce("Rustls production transport origin policy");
    let transport = ReqwestHubTransport::production()
        .expect("the Rustls-only production transport should construct");

    let transport_error = bounded(transport.execute(HubHttpRequest::metadata_get(
        "https://example.invalid/api/models/example/model".to_owned(),
    )))
    .await
    .expect_err("the production transport must reject non-Hugging Face origins before I/O");

    assert!(transport_error.to_string().contains("trusted Hugging Face"));
    let payload_outcome = bounded(transport.execute_payload(HubPayloadRequest::get(
        "https://example.invalid/organization/model/resolve/revision/file".to_owned(),
        0,
    )))
    .await;
    let payload_error = match payload_outcome {
        Err(payload_error) => payload_error,
        Ok(_) => panic!("payload transport must reject an untrusted initial origin before I/O"),
    };
    assert!(payload_error.to_string().contains("trusted Hugging Face"));
}

#[tokio::test]
async fn should_reject_every_consumed_malformed_tree_field_and_digestless_file() {
    announce("tree field and independent digest validation");
    let invalid_entries = [
        serde_json::json!({"type":"file","size":1,"path":"config.json"}),
        serde_json::json!({"type":"file","size":1,"path":"config.json","oid":"not-a-sha"}),
        serde_json::json!({"type":"unknown","size":1,"path":"config.json","oid":GIT_SHA1}),
        serde_json::json!({"type":"file","size":1,"path":"../config.json","oid":GIT_SHA1}),
        serde_json::json!({"type":"file","size":1,"path":"config.json","oid":GIT_SHA1,"xetHash":"bad"}),
        serde_json::json!({"type":"file","size":7,"path":"model.safetensors","oid":GIT_SHA1,"lfs":{"oid":LFS_SHA256,"size":8,"pointerSize":128}}),
        serde_json::json!({"type":"file","size":7,"path":"model.safetensors","oid":GIT_SHA1,"lfs":{"oid":"bad","size":7,"pointerSize":128}}),
        serde_json::json!({"type":"directory","size":0,"path":"weights","lfs":{"oid":LFS_SHA256,"size":0,"pointerSize":128}}),
        serde_json::json!({"type":"directory","size":1,"path":"weights","oid":GIT_SHA1}),
        serde_json::json!({"type":"file","size":9_007_199_254_740_992_u64,"path":"config.json","oid":GIT_SHA1}),
    ];

    for invalid_entry in invalid_entries {
        let hub = hub_with_tree(serde_json::json!([invalid_entry]));
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
            .await
            .expect_err("malformed consumed tree fields must fail closed");
    }
}

#[tokio::test]
async fn should_reject_noncanonical_or_case_colliding_file_paths() {
    announce("portable canonical path and collision validation");
    for invalid_path in [
        "",
        "/config.json",
        "weights//model.bin",
        "weights/./model.bin",
        "weights/../model.bin",
        "weights\\model.bin",
        "modél.bin",
    ] {
        let hub = hub_with_tree(serde_json::json!([
            {"type":"file","size":1,"path":invalid_path,"oid":GIT_SHA1}
        ]));
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
            .await
            .expect_err("noncanonical paths must fail closed");
    }

    let hub = hub_with_tree(serde_json::json!([
        {"type":"file","size":1,"path":"Config.json","oid":GIT_SHA1},
        {"type":"file","size":1,"path":"config.json","oid":GIT_SHA1}
    ]));
    assert!(matches!(
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
        Err(HuggingFaceHubError::CaseInsensitivePathCollision)
    ));
}

#[tokio::test]
async fn should_reject_file_and_descendant_path_conflicts_in_either_tree_order() {
    announce("file and descendant path conflict validation");
    for tree_document in [
        serde_json::json!([
            {"type":"file","size":1,"path":"weights","oid":GIT_SHA1},
            {"type":"file","size":1,"path":"weights/model.bin","oid":GIT_SHA1}
        ]),
        serde_json::json!([
            {"type":"file","size":1,"path":"weights/model.bin","oid":GIT_SHA1},
            {"type":"file","size":1,"path":"weights","oid":GIT_SHA1}
        ]),
    ] {
        let hub = hub_with_tree(tree_document);
        assert!(matches!(
            bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
            Err(HuggingFaceHubError::FilePathHierarchyConflict)
        ));
    }
}

#[tokio::test]
async fn should_reject_a_manifest_that_cannot_fit_in_durable_job_metadata() {
    announce("durable job metadata bound");
    let limits = HuggingFaceHubLimits::default().with_maximum_job_metadata_bytes(4_096);
    let hub = HuggingFaceHub::with_limits(
        scripted_transport([
            valid_metadata_exchange(),
            ScriptedExchange::json(
                tree_url(),
                200,
                serde_json::json!([
                    {"type":"file","size":1,"path":"config.json","oid":GIT_SHA1}
                ]),
            ),
        ]),
        limits,
    );

    assert!(matches!(
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
        Err(HuggingFaceHubError::JobMetadataTooLarge)
    ));
}

#[tokio::test]
async fn should_reject_empty_or_zero_byte_repository_trees() {
    announce("nonempty model payload validation");
    for tree_document in [
        serde_json::json!([]),
        serde_json::json!([
            {"type":"file","size":0,"path":"empty.txt","oid":GIT_SHA1}
        ]),
    ] {
        let hub = hub_with_tree(tree_document);
        assert!(matches!(
            bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
            Err(HuggingFaceHubError::EmptyManifest)
        ));
    }
}

#[tokio::test]
async fn should_enforce_body_page_file_path_and_total_byte_bounds() {
    announce("manifest resource bounds");
    assert!(matches!(
        HubHttpResponse::try_new(200, [], [vec![b'x'; 1_000_001]]),
        Err(HubHttpResponseError::BodyTooLarge)
    ));

    let limits = HuggingFaceHubLimits::new(1, 1, 8, 5);
    let second_page_url = format!(
        "https://huggingface.co/api/models/{REPOSITORY_ID}/tree/{REVISION}?recursive=true&cursor=next"
    );
    let page_limited_hub = HuggingFaceHub::with_limits(
        scripted_transport([
            valid_metadata_exchange(),
            ScriptedExchange::json_with_headers(
                tree_url(),
                200,
                [("link", format!("<{second_page_url}>; rel=\"next\""))],
                serde_json::json!([]),
            ),
        ]),
        limits,
    );
    assert!(matches!(
        bounded(page_limited_hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
        Err(HuggingFaceHubError::TooManyTreePages)
    ));

    for tree_document in [
        serde_json::json!([
            {"type":"file","size":1,"path":"a","oid":GIT_SHA1},
            {"type":"file","size":1,"path":"b","oid":GIT_SHA1}
        ]),
        serde_json::json!([
            {"type":"file","size":1,"path":"too-long!","oid":GIT_SHA1}
        ]),
        serde_json::json!([
            {"type":"file","size":6,"path":"a","oid":GIT_SHA1}
        ]),
    ] {
        let hub = HuggingFaceHub::with_limits(
            scripted_transport([
                valid_metadata_exchange(),
                ScriptedExchange::json(tree_url(), 200, tree_document),
            ]),
            limits,
        );
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION))
            .await
            .expect_err("each configured resource bound must fail closed");
    }
}

#[tokio::test]
async fn should_bound_directory_entries_independently_of_downloadable_files() {
    let limits = HuggingFaceHubLimits::new(1, 1, 32, 100);
    let hub = HuggingFaceHub::with_limits(
        scripted_transport([
            valid_metadata_exchange(),
            ScriptedExchange::json(
                tree_url(),
                200,
                serde_json::json!([
                    {"type":"directory","size":0,"path":"first","oid":GIT_SHA1},
                    {"type":"directory","size":0,"path":"second","oid":GIT_SHA1}
                ]),
            ),
        ]),
        limits,
    );

    assert!(matches!(
        bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
        Err(HuggingFaceHubError::TooManyTreeEntries)
    ));
}

#[tokio::test]
async fn should_reject_pagination_outside_the_exact_hugging_face_tree() {
    announce("pagination origin and immutable tree binding");
    let invalid_next_urls = [
        "http://huggingface.co/api/models/astronomical-test/example-qwen/tree/0123456789abcdef0123456789abcdef01234567?recursive=true",
        "https://example.invalid/api/models/astronomical-test/example-qwen/tree/0123456789abcdef0123456789abcdef01234567?recursive=true",
        "https://huggingface.co.evil.invalid/api/models/astronomical-test/example-qwen/tree/0123456789abcdef0123456789abcdef01234567?recursive=true",
        "https://huggingface.co/api/models/astronomical-test/example-qwen/tree/1111111111111111111111111111111111111111?recursive=true",
        "https://huggingface.co/api/models/astronomical-test/example-qwen/tree/0123456789abcdef0123456789abcdef01234567?cursor=next",
    ];

    for invalid_next_url in invalid_next_urls {
        let hub = HuggingFaceHub::new(scripted_transport([
            valid_metadata_exchange(),
            ScriptedExchange::json_with_headers(
                tree_url(),
                200,
                [("link", format!("<{invalid_next_url}>; rel=\"next\""))],
                serde_json::json!([]),
            ),
        ]));
        assert!(matches!(
            bounded(hub.fetch_manifest(REPOSITORY_ID, REVISION)).await,
            Err(HuggingFaceHubError::UnsafePaginationLink)
        ));
    }
}

async fn bounded<T>(operation: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(2), operation)
        .await
        .expect("the hermetic Hub operation should finish within two seconds")
}
