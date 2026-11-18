//! Strict bundled catalog contracts: the shipped file parses, and malformed,
//! oversized or unsafe documents and entries are refused.

use crate::{DownloadCatalog, DownloadCatalogEntry, DownloadCatalogRole};

const VALID_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// Guards that the shipped, compiled-in catalog parses, validates, and is non-empty.
/// `include_str!` only proves the file exists at compile time; this proves its contents
/// are a valid catalog in the cheap integration lane, not only at e2e startup time.
#[test]
fn should_parse_and_load_the_shipped_bundled_catalog() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    assert!(
        !download_catalog.entries().is_empty(),
        "the shipped catalog must offer at least one model"
    );
}

#[test]
fn should_accept_complete_entries_for_all_executable_families_in_authored_order() {
    let catalog_json = serde_json::json!({
        "schema_version": 5,
        "entries": [
            valid_entry("astronomical-test/example-qwen", "qwen3_5"),
            valid_entry("astronomical-test/example-flux", "flux2_klein"),
            valid_entry("astronomical-test/example-embedder", "modernbert"),
            valid_entry("astronomical-test/example-prism", "prism_hadamard_qwen35"),
            valid_entry("astronomical-test/example-decider", "laya"),
        ]
    });

    let download_catalog = DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect("complete fictional entries should be accepted");

    assert_eq!(download_catalog.entries().len(), 5);
    assert_eq!(
        download_catalog.entries()[4].huggingface_id(),
        "astronomical-test/example-decider"
    );
    assert_eq!(
        download_catalog.entries()[0].huggingface_id(),
        "astronomical-test/example-qwen"
    );
    assert_eq!(
        download_catalog.entries()[1].huggingface_id(),
        "astronomical-test/example-flux"
    );
    assert_eq!(
        download_catalog.entries()[2].huggingface_id(),
        "astronomical-test/example-embedder"
    );
    assert_eq!(
        download_catalog.entries()[3].huggingface_id(),
        "astronomical-test/example-prism"
    );
}

#[test]
fn should_accept_a_decision_entry_and_carry_its_capability() {
    let decision_entry = entry_with_overrides(serde_json::json!({
        "family": "laya",
        "role": "decision",
        "capabilities": {"supports_decisions": true}
    }));

    let download_catalog =
        DownloadCatalog::parse_json(&catalog_with_entry(decision_entry).to_string())
            .expect("a decision entry should be accepted");

    let catalog_entry = &download_catalog.entries()[0];
    assert_eq!(catalog_entry.role(), DownloadCatalogRole::Decision);
    assert_eq!(catalog_entry.role().as_str(), "decision");
    assert!(catalog_entry.capabilities().supports_decisions);
    assert!(!catalog_entry.capabilities().supports_embeddings);
}

/// The shipped Laya entry downloads the two checkpoints whose answers were
/// compared with the reference implementation, and nothing else from the
/// repository.
#[test]
fn should_ship_laya_as_a_decision_model_with_only_its_checked_checkpoints() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let mut laya_entry = None;
    for catalog_entry in download_catalog.entries() {
        if catalog_entry.huggingface_id() == "cavi-ai/laya-MLX-8bit" {
            laya_entry = Some(catalog_entry);
        }
    }
    let laya_entry = laya_entry.expect("the shipped catalog should offer Laya");

    assert_eq!(laya_entry.role(), DownloadCatalogRole::Decision);
    assert!(laya_entry.capabilities().supports_decisions);
    assert_eq!(laya_entry.upstream_license(), Some("Apache-2.0"));
    let selection = laya_entry.download_path_selection();
    assert!(selection.includes("model.safetensors"));
    assert!(selection.includes("multilingual/model.safetensors"));
    assert!(!selection.includes("typed-decisions/model.safetensors"));
    assert!(!selection.includes("laya/laya.py"));
    let description = laya_entry
        .description()
        .expect("the Laya entry should state its limits");
    assert!(description.contains("512 tokens") && description.contains("1,024 tokens"));
}

/// The shipped Clef entry downloads the backbone, the joint head and the
/// tokenizer the reference comparison used, and not the build's own Python.
#[test]
fn should_ship_clef_flash_as_a_decision_model_with_its_head_and_limit() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let mut clef_entry = None;
    for catalog_entry in download_catalog.entries() {
        if catalog_entry.huggingface_id() == "mlx-community/clef-flash-8bit" {
            clef_entry = Some(catalog_entry);
        }
    }
    let clef_entry = clef_entry.expect("the shipped catalog should offer Clef Flash");

    assert_eq!(clef_entry.role(), DownloadCatalogRole::Decision);
    assert_eq!(clef_entry.family().as_str(), "clef");
    assert!(clef_entry.capabilities().supports_decisions);
    assert_eq!(clef_entry.upstream_license(), Some("Apache-2.0"));
    let selection = clef_entry.download_path_selection();
    assert!(selection.includes("joint_head.safetensors"));
    assert!(selection.includes("joint_head_config.json"));
    assert!(selection.includes("model-00002-of-00002.safetensors"));
    assert!(!selection.includes("clef_mlx.py"));
    let description = clef_entry
        .description()
        .expect("the Clef entry should state its limits");
    assert!(description.contains("16,384 tokens") && description.contains("Text only"));
}

/// Clef's memory figure is the peak `cargo measure-clef-decision-latency`
/// printed after requests at the 16,384-token limit, not an estimate from
/// the weight files. It is more than the files, which a forward pass works
/// beside.
#[test]
fn should_state_clef_flash_memory_as_the_peak_measured_at_its_token_limit() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let mut clef_memory_bytes = None;
    let mut clef_size_bytes = None;
    for catalog_entry in download_catalog.entries() {
        if catalog_entry.huggingface_id() == "mlx-community/clef-flash-8bit" {
            clef_memory_bytes = Some(catalog_entry.approximate_memory_bytes());
            clef_size_bytes = Some(catalog_entry.approximate_size_bytes());
        }
    }

    assert_eq!(clef_memory_bytes, Some(12_380_644_404));
    assert!(clef_memory_bytes > clef_size_bytes);
}

#[test]
fn should_accept_bounded_executable_payload_selection_and_reject_unsafe_rules() {
    let selected_entry = entry_with_overrides(serde_json::json!({
        "included_paths": ["LICENSE.md", "model_index.json", "transformer/"]
    }));
    let download_catalog =
        DownloadCatalog::parse_json(&catalog_with_entry(selected_entry).to_string())
            .expect("release-authored files and directory prefixes should be selectable");
    let selection = download_catalog.entries()[0].download_path_selection();
    assert!(selection.includes("LICENSE.md"));
    assert!(selection.includes("transformer/model.safetensors"));
    assert!(!selection.includes("alternate-model.safetensors"));

    for invalid_paths in [
        serde_json::json!([]),
        serde_json::json!(["../weights"]),
        serde_json::json!(["weights\\model.safetensors"]),
        serde_json::json!(["weights/", "weights/model.safetensors"]),
        serde_json::json!(["Transformer/", "transformer/"]),
    ] {
        let invalid_entry = entry_with_overrides(serde_json::json!({
            "included_paths": invalid_paths
        }));
        DownloadCatalog::parse_json(&catalog_with_entry(invalid_entry).to_string())
            .expect_err("unsafe, empty, colliding, or redundant selections must fail closed");
    }
}

#[test]
fn should_reject_invalid_catalog_document_shapes() {
    for invalid_catalog_json in [
        r#"{"entries":[]}"#,
        r#"{"schema_version":1,"entries":[]}"#,
        r#"{"schema_version":5}"#,
        r#"{"schema_version":5,"entries":{}}"#,
        r#"{"schema_version":5,"entries":[],"unexpected":true}"#,
        r#"{"schema_version":5,"schema_version":5,"entries":[]}"#,
        r#"{"schema_version":5,"entries":[]} trailing"#,
    ] {
        DownloadCatalog::parse_json(invalid_catalog_json)
            .expect_err("an invalid catalog document shape must fail closed");
    }
}

#[test]
fn should_reject_catalog_metadata_that_exceeds_startup_resource_bounds() {
    let oversized_catalog_document = " ".repeat(1_000_001);
    DownloadCatalog::parse_json(&oversized_catalog_document)
        .expect_err("catalog metadata beyond the startup byte bound must fail closed");

    let oversized_entry_count_catalog = serde_json::json!({
        "schema_version": 5,
        "entries": (0..1_025)
            .map(|entry_index| valid_entry(
                &format!("astronomical-test/example-{entry_index}"),
                "qwen3_5",
            ))
            .collect::<Vec<_>>()
    });
    DownloadCatalog::parse_json(&oversized_entry_count_catalog.to_string())
        .expect_err("catalog entry counts beyond the startup bound must fail closed");
}

#[test]
fn should_reject_unsafe_or_non_hub_repository_identities() {
    for invalid_huggingface_id in [
        "",
        "single-component",
        "/model",
        "organization/",
        "organization/model/extra",
        " organization/model",
        "organization/model ",
        "organization/../model",
        "organization/model\\child",
        "organization/.model",
        "organization/model-",
        "organization/model--variant",
        "organization/model..variant",
        "organization/model.git",
        "organization/model name",
        "organization/modél",
    ] {
        let catalog_json = catalog_with_entry(valid_entry(invalid_huggingface_id, "qwen3_5"));
        DownloadCatalog::parse_json(&catalog_json.to_string())
            .expect_err("an unsafe or unsupported repository identity must fail closed");
    }

    let oversized_model_name = "m".repeat(97);
    let catalog_json = catalog_with_entry(valid_entry(
        &format!("astronomical-test/{oversized_model_name}"),
        "qwen3_5",
    ));
    DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect_err("a repository name beyond the Hub bound must fail closed");
}

#[test]
fn should_reject_invalid_revision_family_visibility_and_size() {
    let invalid_entries = [
        entry_with_overrides(serde_json::json!({"revision": "main"})),
        entry_with_overrides(serde_json::json!({
            "revision": "0123456789abcdef0123456789abcdef0123456g"
        })),
        entry_with_overrides(serde_json::json!({
            "revision": "0123456789abcdef0123456789abcdef0123456A"
        })),
        entry_with_overrides(serde_json::json!({"family": "unsupported_family"})),
        entry_with_overrides(serde_json::json!({"public": false})),
        entry_with_overrides(serde_json::json!({"approximate_size_bytes": 0})),
        entry_with_overrides(
            serde_json::json!({"approximate_size_bytes": 9_007_199_254_740_992_u64}),
        ),
    ];
    for invalid_entry in invalid_entries {
        let catalog_json = catalog_with_entry(invalid_entry);
        DownloadCatalog::parse_json(&catalog_json.to_string())
            .expect_err("invalid entry metadata must fail closed");
    }

    DownloadCatalog::parse_json(&format!(
        r#"{{"schema_version":5,"entries":[{{"huggingface_id":"astronomical-test/example-qwen","revision":"{VALID_REVISION}","family":"qwen3_5","approximate_size_bytes":4000000000,"approximate_memory_bytes":2000000000,"role":"chat"}}]}}"#
    ))
    .expect_err("public must remain a required field");
}

#[test]
fn should_accept_the_largest_approximate_size_that_the_library_can_render_exactly() {
    let catalog_json = catalog_with_entry(entry_with_overrides(
        serde_json::json!({"approximate_size_bytes": 9_007_199_254_740_991_u64}),
    ));

    let download_catalog = DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect("the maximum exactly renderable byte count should remain valid");

    assert_eq!(
        download_catalog.entries()[0].approximate_size_bytes(),
        9_007_199_254_740_991
    );
}

#[test]
fn should_reject_unknown_duplicate_and_case_colliding_entries() {
    let extra_field_entry = entry_with_overrides(serde_json::json!({"unexpected": true}));
    DownloadCatalog::parse_json(&catalog_with_entry(extra_field_entry).to_string())
        .expect_err("unknown entry fields must fail closed");

    let duplicated_entry = valid_entry("astronomical-test/example-qwen", "qwen3_5");
    let duplicate_catalog = serde_json::json!({
        "schema_version": 5,
        "entries": [duplicated_entry.clone(), duplicated_entry]
    });
    DownloadCatalog::parse_json(&duplicate_catalog.to_string())
        .expect_err("exact duplicate identities must fail closed");

    let case_collision_catalog = serde_json::json!({
        "schema_version": 5,
        "entries": [
            valid_entry("Astronomical-Test/Example-Qwen", "qwen3_5"),
            valid_entry("astronomical-test/example-qwen", "qwen3_5"),
        ]
    });
    DownloadCatalog::parse_json(&case_collision_catalog.to_string())
        .expect_err("case-insensitive destination collisions must fail closed");
}

fn valid_entry(huggingface_id: &str, family: &str) -> serde_json::Value {
    serde_json::json!({
        "huggingface_id": huggingface_id,
        "revision": VALID_REVISION,
        "family": family,
        "approximate_size_bytes": 4_000_000_000_u64,
        "approximate_memory_bytes": 2_000_000_000_u64,
        "role": "chat",
        "public": true,
    })
}

fn entry_with_overrides(overrides: serde_json::Value) -> serde_json::Value {
    let mut catalog_entry = valid_entry("astronomical-test/example-qwen", "qwen3_5");
    for (field_name, field_value) in overrides
        .as_object()
        .expect("entry overrides should be an object")
    {
        catalog_entry
            .as_object_mut()
            .expect("the valid entry should be an object")
            .insert(field_name.clone(), field_value.clone());
    }
    catalog_entry
}

fn catalog_with_entry(catalog_entry: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"schema_version": 5, "entries": [catalog_entry]})
}

/// Counts the Get started picks for one role so catalog-wide uniqueness tests
/// can state their expectation without a closure.
fn count_get_started_picks_for(
    download_catalog: &DownloadCatalog,
    role: DownloadCatalogRole,
) -> usize {
    let mut pick_count = 0;
    for catalog_entry in download_catalog.entries() {
        if catalog_entry.get_started() && catalog_entry.role() == role {
            pick_count += 1;
        }
    }
    pick_count
}

#[test]
fn should_offer_exactly_one_get_started_pick_for_chat_coding_and_image_generation() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");

    assert_eq!(
        count_get_started_picks_for(&download_catalog, DownloadCatalogRole::Chat),
        1
    );
    assert_eq!(
        count_get_started_picks_for(&download_catalog, DownloadCatalogRole::Coding),
        1
    );
    assert_eq!(
        count_get_started_picks_for(&download_catalog, DownloadCatalogRole::ImageGeneration),
        1
    );
}

#[test]
fn should_reject_an_unknown_role() {
    let catalog_json = catalog_with_entry(entry_with_overrides(
        serde_json::json!({"role": "translation"}),
    ));
    DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect_err("a role outside the closed role set must fail closed");
}

#[test]
fn should_reject_two_get_started_picks_for_one_role_across_the_catalog() {
    let catalog_json = serde_json::json!({
        "schema_version": 5,
        "entries": [
            entry_with_overrides(serde_json::json!({
                "huggingface_id": "astronomical-test/first-chat",
                "role": "chat",
                "get_started": true,
            })),
            entry_with_overrides(serde_json::json!({
                "huggingface_id": "astronomical-test/second-chat",
                "role": "chat",
                "get_started": true,
            })),
        ]
    });
    DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect_err("two Get started picks for one role must fail closed");
}

#[test]
fn should_accept_one_get_started_pick_per_role_across_the_catalog() {
    let catalog_json = serde_json::json!({
        "schema_version": 5,
        "entries": [
            entry_with_overrides(serde_json::json!({
                "huggingface_id": "astronomical-test/chat-pick",
                "role": "chat",
                "get_started": true,
            })),
            entry_with_overrides(serde_json::json!({
                "huggingface_id": "astronomical-test/coding-pick",
                "role": "coding",
                "get_started": true,
            })),
            entry_with_overrides(serde_json::json!({
                "huggingface_id": "astronomical-test/image-pick",
                "role": "image_generation",
                "get_started": true,
            })),
        ]
    });
    let download_catalog = DownloadCatalog::parse_json(&catalog_json.to_string())
        .expect("one Get started pick per role should be accepted");
    assert_eq!(download_catalog.entry_count(), 3);
}

/// The shipped speech entries download only what their engines read: Kokoro
/// its weights and its American English voices (not the `.pt` copies or the
/// other languages' voices), Phonon its weights, tokenizer and voice. Neither
/// is experimental: both speak through the served endpoint, and Kokoro's
/// arithmetic is held against its PyTorch reference.
#[test]
fn should_ship_kokoro_and_phonon_as_speech_models() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let speech_entries: Vec<_> = download_catalog
        .entries()
        .iter()
        .filter(|catalog_entry| catalog_entry.role() == DownloadCatalogRole::Speech)
        .collect();
    assert_eq!(speech_entries.len(), 2);
    for speech_entry in &speech_entries {
        assert!(speech_entry.capabilities().supports_speech);
        assert!(!speech_entry.experimental());
        assert_eq!(speech_entry.upstream_license(), Some("Apache-2.0"));
    }
    let kokoro_entry = speech_entries
        .iter()
        .find(|catalog_entry| catalog_entry.huggingface_id() == "mlx-community/Kokoro-82M-bf16")
        .expect("the shipped catalog should offer Kokoro");
    let kokoro_selection = kokoro_entry.download_path_selection();
    assert!(kokoro_selection.includes("voices/af_heart.safetensors"));
    assert!(!kokoro_selection.includes("voices/af_heart.pt"));
    assert!(!kokoro_selection.includes("voices/bf_emma.safetensors"));
    let phonon_entry = speech_entries
        .iter()
        .find(|catalog_entry| catalog_entry.huggingface_id() == "gradium/phonon-31m-en")
        .expect("the shipped catalog should offer Phonon");
    assert!(
        phonon_entry
            .download_path_selection()
            .includes("default-voice.safetensors")
    );
    assert!(
        !phonon_entry
            .download_path_selection()
            .includes("model.q8.gguf")
    );
}

/// The plain 4-bit Qwen 3.8 27B is the target the DFlash2 drafter pairs with,
/// so the catalog offers it as an ordinary chat model with its vision tower
/// and every file in the repository: nothing is filtered out, because the
/// processor and chat template files are what make pictures and tools work.
#[test]
fn should_ship_the_plain_four_bit_qwen3_8_27b_as_a_vision_chat_model() {
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let qwen3_8_entry = download_catalog
        .entries()
        .iter()
        .find(|catalog_entry| catalog_entry.huggingface_id() == "TensorFold/Qwen3.8-27B-MLX-4bit")
        .expect("the shipped catalog should offer the plain 4-bit Qwen 3.8 27B");

    assert_eq!(qwen3_8_entry.role(), DownloadCatalogRole::Chat);
    assert_eq!(qwen3_8_entry.family().as_str(), "qwen3_5");
    assert_eq!(
        qwen3_8_entry.revision(),
        "22d8d538154e0e5b6f8dcb1fcc74a60a0798104c"
    );
    assert!(!qwen3_8_entry.experimental() && !qwen3_8_entry.get_started());
    let capabilities = qwen3_8_entry.capabilities();
    assert!(capabilities.supports_vision && capabilities.supports_reasoning);
    assert!(capabilities.supports_tool_calls);
    assert!(qwen3_8_entry.approximate_memory_bytes() > qwen3_8_entry.approximate_size_bytes());
    let selection = qwen3_8_entry.download_path_selection();
    assert!(selection.includes("model-00003-of-00003.safetensors"));
    assert!(selection.includes("preprocessor_config.json"));
    assert!(selection.includes("chat_template.jinja"));
}

/// A DFlash2 package is a plain catalog model with its DFlash2 drafter in a
/// `drafter` folder, which the engine pairs with the model it sits inside. The
/// download must therefore keep that folder, and the size the disk check
/// reserves must count the drafter on top of its plain twin. When the drafter
/// cannot load (no M5 GPU, or a memory limit with no room for it) the model
/// decodes exactly like its twin, so it streams from the same memory floor.
#[test]
fn should_ship_the_optiq_dflash2_package_with_its_bundled_drafter() {
    assert_dflash2_package_matches_its_plain_twin(DFlash2PackageExpectation {
        package_huggingface_id: "ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-DFLASH2-4bit",
        plain_twin_huggingface_id: "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit",
        pinned_revision: "28c5203a83fa7bc39b3363b870273c23f6bcf04e",
        vision_tower_path: "optiq/optiq_vision.safetensors",
    });
}

#[test]
fn should_ship_the_8bit_vision_dflash2_package_with_its_bundled_drafter() {
    assert_dflash2_package_matches_its_plain_twin(DFlash2PackageExpectation {
        package_huggingface_id: "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision-DFLASH2",
        plain_twin_huggingface_id: "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision",
        pinned_revision: "73cb28bdf5ad151ddddefdfd5e4635117e1fbef7",
        vision_tower_path: "model-vision.safetensors",
    });
}

struct DFlash2PackageExpectation {
    package_huggingface_id: &'static str,
    plain_twin_huggingface_id: &'static str,
    pinned_revision: &'static str,
    vision_tower_path: &'static str,
}

fn assert_dflash2_package_matches_its_plain_twin(expectation: DFlash2PackageExpectation) {
    const DRAFTER_WEIGHTS_BYTES: u64 = 1_052_513_488;
    let download_catalog =
        DownloadCatalog::load_bundled().expect("the shipped bundled catalog should parse");
    let package_entry = shipped_entry(&download_catalog, expectation.package_huggingface_id);
    let plain_twin_entry = shipped_entry(&download_catalog, expectation.plain_twin_huggingface_id);

    assert_eq!(package_entry.role(), plain_twin_entry.role());
    assert_eq!(package_entry.family().as_str(), "qwen3_5");
    assert_eq!(package_entry.revision(), expectation.pinned_revision);
    assert!(!package_entry.experimental() && !package_entry.get_started());
    assert_eq!(
        package_entry.capabilities(),
        plain_twin_entry.capabilities()
    );
    let selection = package_entry.download_path_selection();
    assert!(selection.includes("drafter/model.safetensors"));
    assert!(selection.includes("drafter/config.json"));
    assert!(selection.includes(expectation.vision_tower_path));
    assert!(
        package_entry.approximate_size_bytes()
            >= plain_twin_entry.approximate_size_bytes() + DRAFTER_WEIGHTS_BYTES
    );
    assert_eq!(
        package_entry.approximate_memory_bytes() - package_entry.approximate_size_bytes(),
        plain_twin_entry.approximate_memory_bytes() - plain_twin_entry.approximate_size_bytes()
    );
    assert_eq!(
        package_entry.minimum_memory_bytes(),
        plain_twin_entry.minimum_memory_bytes()
    );
}

fn shipped_entry<'catalog>(
    download_catalog: &'catalog DownloadCatalog,
    huggingface_id: &str,
) -> &'catalog DownloadCatalogEntry {
    download_catalog
        .entries()
        .iter()
        .find(|catalog_entry| catalog_entry.huggingface_id() == huggingface_id)
        .unwrap_or_else(|| panic!("the shipped catalog should offer {huggingface_id}"))
}
