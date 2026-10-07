import Foundation

import Testing

import JourneyCategories

@testable import Supervisor

/**
 * Acceptance journeys for the download catalog contract, migrating the
 * validation behavior of apps/supervisor/src/library/download_catalog.rs and
 * download_path_selection.rs: the catalog parses strictly in authored order,
 * rejects unsafe or colliding metadata, keeps every entry public and
 * immutably revised, and the release-bundled document always validates.
 */
@Suite(.tags(.hermeticJourney))
final class DownloadCatalogTests {

    static let VALID_CATALOG_JSON: String = """
    {
        "schema_version": 2,
        "entries": [
            {
                "huggingface_id": "astronomical-test/example-qwen",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "display_name": "Example Qwen",
                "family": "qwen3_5",
                "approximate_size_bytes": 4000000000,
                "public": true
            },
            {
                "huggingface_id": "astronomical-test/example-embedder",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
                "display_name": "Example Embedder",
                "family": "modernbert",
                "approximate_size_bytes": 200000000,
                "public": true,
                "capabilities": {"supports_embeddings": true}
            }
        ]
    }
    """

    @Test
    func should_parse_the_validated_catalog_in_authored_order() throws {
        let downloadCatalog: DownloadCatalog = try DownloadCatalog.parseJson(DownloadCatalogTests.VALID_CATALOG_JSON)

        #expect(downloadCatalog.schemaVersion == 2)
        #expect(downloadCatalog.entryCount == 2)
        #expect(downloadCatalog.entries[0].huggingfaceId == "astronomical-test/example-qwen")
        #expect(downloadCatalog.entries[1].huggingfaceId == "astronomical-test/example-embedder")
        #expect(downloadCatalog.entries[0].family == DownloadCatalogFamily.qwen3_5)
        #expect(downloadCatalog.entries[1].family == DownloadCatalogFamily.modernbert)
        #expect(downloadCatalog.entries[0].approximateSizeBytes == 4_000_000_000)
        #expect(downloadCatalog.entries[1].capabilities.supportsEmbeddings == true)
        #expect(downloadCatalog.entries[0].capabilities.supportsEmbeddings == false)
        #expect(downloadCatalog.entries[0].description == nil)
        #expect(downloadCatalog.entries[0].downloadPathSelection.includes(relativePath: "config.json"))
    }

    @Test
    func should_reject_an_unsupported_schema_version_and_unknown_fields() throws {
        let unsupportedSchemaVersion: DownloadCatalogError? = DownloadCatalogTests.parseError(
            "{\"schema_version\":1,\"entries\":[]}")
        #expect(unsupportedSchemaVersion == DownloadCatalogError.unsupportedSchemaVersion(schemaVersion: 1))

        let unknownEntryField: DownloadCatalogError? = DownloadCatalogTests.parseError(
            "{\"schema_version\":2,\"entries\":[{\"huggingface_id\":\"a/b\",\"revision\":\""
                + "0123456789abcdef0123456789abcdef01234567"
                + "\",\"display_name\":\"x\",\"family\":\"qwen3_5\",\"approximate_size_bytes\":1,\"public\":true,\"surprise\":1}]}")
        #expect(unknownEntryField == DownloadCatalogError.parse)
    }

    @Test
    func should_reject_duplicate_and_case_colliding_identities() throws {
        for collidingSecondId: String in ["astronomical-test/example-qwen", "Astronomical-Test/Example-Qwen"] {
            let duplicateError: DownloadCatalogError? = DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":["
                    + DownloadCatalogTests.entryJson(huggingfaceId: "astronomical-test/example-qwen")
                    + "," + DownloadCatalogTests.entryJson(huggingfaceId: collidingSecondId)
                    + "]}")
            #expect(duplicateError == DownloadCatalogError.duplicateHuggingFaceId)
        }
    }

    @Test
    func should_reject_invalid_identities_revisions_and_visibility() throws {
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(huggingfaceId: "single-component") + "]}")
                == DownloadCatalogError.invalidHuggingFaceId(entryIndex: 0))
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(huggingfaceId: "org/model.git") + "]}")
                == DownloadCatalogError.invalidHuggingFaceId(entryIndex: 0))
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(revision: "not-a-commit-sha") + "]}")
                == DownloadCatalogError.invalidRevision(entryIndex: 0))
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(revision: "0123456789ABCDEF0123456789ABCDEF01234567") + "]}")
                == DownloadCatalogError.invalidRevision(entryIndex: 0))
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(isPublic: false) + "]}")
                == DownloadCatalogError.modelNotPublic(entryIndex: 0))
        #expect(
            DownloadCatalogTests.parseError(
                "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(approximateSizeBytes: 0) + "]}")
                == DownloadCatalogError.invalidApproximateSize(entryIndex: 0))
    }

    @Test
    func should_reject_empty_capability_objects() throws {
        let emptyCapabilitiesError: DownloadCatalogError? = DownloadCatalogTests.parseError(
            "{\"schema_version\":2,\"entries\":[" + DownloadCatalogTests.entryJson(extraJson: ",\"capabilities\":{}") + "]}")
        #expect(emptyCapabilitiesError == DownloadCatalogError.invalidCapabilities(entryIndex: 0))
    }

    @Test
    func should_reject_overlapping_included_paths() throws {
        let overlappingPathsError: DownloadCatalogError? = DownloadCatalogTests.parseError(
            "{\"schema_version\":2,\"entries\":["
                + DownloadCatalogTests.entryJson(extraJson: ",\"included_paths\":[\"weights/\",\"weights/model.safetensors\"]") + "]}")
        #expect(overlappingPathsError == DownloadCatalogError.invalidIncludedPaths(entryIndex: 0))

        let escapingPathsError: DownloadCatalogError? = DownloadCatalogTests.parseError(
            "{\"schema_version\":2,\"entries\":["
                + DownloadCatalogTests.entryJson(extraJson: ",\"included_paths\":[\"../outside\"]") + "]}")
        #expect(escapingPathsError == DownloadCatalogError.invalidIncludedPaths(entryIndex: 0))
    }

    @Test
    func should_keep_an_included_path_selection_case_insensitive() throws {
        let selection: DownloadPathSelection = try DownloadPathSelection(includedPaths: ["Weights/Model.safetensors", "Docs/"])

        #expect(selection.includes(relativePath: "weights/model.safetensors"))
        #expect(selection.includes(relativePath: "docs/anything.txt"))
        #expect(!selection.includes(relativePath: "weights/other.safetensors"))
    }

    @Test
    func should_load_and_validate_the_release_bundled_catalog() throws {
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled()

        #expect(bundledCatalog.schemaVersion == 2)
        #expect(bundledCatalog.entryCount > 0)
        let bundledIdentityCount: Int = Set(bundledCatalog.entries.map { (entry: DownloadCatalogEntry) -> String in
            return entry.huggingfaceId.lowercased()
        }).count
        #expect(bundledIdentityCount == bundledCatalog.entryCount, "bundled identities never collide")
    }

    @Test
    func should_offer_qwen_3_6_35b_a3b_optiq_4bit_from_the_bundled_catalog() throws {
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        let qwenEntry: DownloadCatalogEntry = try #require(
            bundledCatalog.entries.first(where: { (entry: DownloadCatalogEntry) -> Bool in
                return entry.huggingfaceId == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit";
            }),
            "the release catalog should include the Qwen 3.6 35B-A3B OptiQ 4-bit offer");

        // The pinned revision is the contract: it must match the reviewed Hub
        // tree, not whatever the repository head happens to be.
        #expect(qwenEntry.revision == "70a3aa32c7feef511182bf16aa332f37e8d82014");
        #expect(qwenEntry.family == .qwen3_5);
        #expect(qwenEntry.approximateSizeBytes == 24_693_956_069);
        #expect(qwenEntry.upstreamLicense == "Apache-2.0");
        #expect(qwenEntry.quantizationLabel == "OptiQ mixed-precision 4/8-bit (affine, group 64)");
        #expect(qwenEntry.capabilities.supportsReasoning == true);
        #expect(qwenEntry.capabilities.supportsVision == true);
        #expect(qwenEntry.capabilities.supportsToolCalls == true);
        #expect(qwenEntry.capabilities.supportsEmbeddings == false);
        #expect(qwenEntry.capabilities.supportsImageGeneration == false);
        #expect(qwenEntry.capabilities.contextWindow == 262_144);

        let pathSelection: DownloadPathSelection = qwenEntry.downloadPathSelection;
        #expect(pathSelection.includes(relativePath: "config.json"));
        #expect(pathSelection.includes(relativePath: "model.safetensors.index.json"));
        #expect(pathSelection.includes(relativePath: "model-00001-of-00005.safetensors"));
        #expect(pathSelection.includes(relativePath: "model-00005-of-00005.safetensors"));
        #expect(pathSelection.includes(relativePath: "optiq/optiq_vision.safetensors"));
        #expect(pathSelection.includes(relativePath: "optiq/mtp.safetensors"));
        #expect(pathSelection.includes(relativePath: "tokenizer.json"));
    }

    @Test
    func should_offer_the_stacked_affine_k2_horizon_mova_variant_from_the_bundled_catalog() throws {
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        let k2Entries: Array<DownloadCatalogEntry> = bundledCatalog.entries.filter { (entry: DownloadCatalogEntry) -> Bool in
            return entry.family == .k2HorizonMoVA;
        };
        #expect(k2Entries.count == 1, "the catalog offers one executable K2 Horizon MoVA conversion");
        let k2Entry: DownloadCatalogEntry = try #require(k2Entries.first);

        #expect(k2Entry.huggingfaceId == "abenzerps/K2-Horizon-MoVA-36B-A4B-MLX-4bit");
        #expect(k2Entry.revision == "0c576733b69e8ca2d7a0292d0f01ee1d955bb9b5");
        #expect(k2Entry.approximateSizeBytes == 21_102_192_601);
        #expect(k2Entry.upstreamLicense == "Apache-2.0");
        #expect(k2Entry.quantizationLabel == "4-bit affine (group 64)");
        #expect(k2Entry.capabilities.supportsReasoning == true);
        #expect(k2Entry.capabilities.supportsToolCalls == true);
        #expect(k2Entry.capabilities.supportsVision == false);
        #expect(k2Entry.capabilities.supportsEmbeddings == false);
        #expect(k2Entry.capabilities.supportsImageGeneration == false);
        #expect(k2Entry.capabilities.contextWindow == 524_288);

        let pathSelection: DownloadPathSelection = k2Entry.downloadPathSelection;
        #expect(pathSelection.includes(relativePath: "config.json"));
        #expect(pathSelection.includes(relativePath: "model.safetensors.index.json"));
        #expect(pathSelection.includes(relativePath: "tokenizer.json"));
        #expect(pathSelection.includes(relativePath: "chat_template.jinja"));
        #expect(pathSelection.includes(relativePath: "model-00001-of-00048.safetensors"));
        #expect(pathSelection.includes(relativePath: "model-00048-of-00048.safetensors"));
        #expect(pathSelection.includes(relativePath: "README.md") == false,
            "the executable payload must not ship repository extras");
        #expect(pathSelection.includes(relativePath: "k2_horizon_mova_mlx.py") == false,
            "the executable payload must not ship checkpoint Python");
        #expect(pathSelection.includes(relativePath: "assets/k2-horizon-mova-36b-a4b-benchmarks.png") == false,
            "the executable payload must not ship marketing assets");
    }

    @Test
    func should_offer_ornith_1_5_35b_a3b_mlx_6bit_from_the_bundled_catalog() throws {
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        let ornithEntry: DownloadCatalogEntry = try #require(
            bundledCatalog.entries.first(where: { (entry: DownloadCatalogEntry) -> Bool in
                return entry.huggingfaceId == "ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit";
            }),
            "the release catalog should include the Ornith MLX 6-bit offer");

        #expect(ornithEntry.revision == "585b7867b0517980293ece857b26d64e84491352");
        #expect(ornithEntry.family == .qwen3_5);
        #expect(ornithEntry.upstreamLicense == "MIT");
        #expect(ornithEntry.quantizationLabel == "6-bit affine (group 64)");
        let architectureSummary: String = try #require(ornithEntry.architectureSummary);
        #expect(architectureSummary.contains("256 experts") == true);
        #expect(architectureSummary.contains("8 routed per token") == true);
        #expect(ornithEntry.capabilities.supportsReasoning == true);
        #expect(ornithEntry.capabilities.supportsToolCalls == true);
        #expect(ornithEntry.capabilities.supportsVision == false);
        #expect(ornithEntry.capabilities.supportsEmbeddings == false);
        #expect(ornithEntry.capabilities.supportsImageGeneration == false);
        #expect(ornithEntry.capabilities.contextWindow == 262_144);

        let pathSelection: DownloadPathSelection = ornithEntry.downloadPathSelection;
        for shardIndex in 1...6 {
            #expect(pathSelection.includes(relativePath: String(
                format: "model-%05d-of-00006.safetensors",
                shardIndex)));
        }
        for requiredPath in [
            "config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
        ] {
            #expect(pathSelection.includes(relativePath: requiredPath), Comment(stringLiteral: requiredPath));
        }
    }

    @Test
    func should_package_qwen_image_2_1_with_only_its_executable_pipeline_graph() throws {
        let bundledCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        let qwenImageEntry: DownloadCatalogEntry = try #require(
            bundledCatalog.entries.first(where: { (entry: DownloadCatalogEntry) -> Bool in
                return entry.family == .qwenImage21;
            }),
            "the release catalog should include the executable Qwen-Image-2.1 profile");

        // The pinned revision is the contract: it must match the reviewed
        // 4-bit MLX conversion the acceptance journeys render with, not
        // whatever the repository head happens to be.
        #expect(qwenImageEntry.huggingfaceId == "mlx-community/Qwen-Image-2.1-MLX-4bit");
        #expect(qwenImageEntry.revision == "4db4e8c0c0e7a1debf0320415bec8388e888494c");
        #expect(qwenImageEntry.capabilities.supportsImageGeneration == true);
        #expect(qwenImageEntry.capabilities.supportsReasoning == false);
        #expect(qwenImageEntry.capabilities.supportsEmbeddings == false);
        // Keep the authored size within a decimal band so re-quantizations do
        // not silently invalidate the preflight disk estimate.
        #expect(qwenImageEntry.approximateSizeBytes > 9_000_000_000);
        #expect(qwenImageEntry.approximateSizeBytes < 12_000_000_000);

        let pathSelection: DownloadPathSelection = qwenImageEntry.downloadPathSelection;
        for requiredPath in [
            "model_index.json",
            "processor/tokenizer.json",
            "processor/preprocessor_config.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "transformer/config.json",
            "transformer/model.safetensors",
            "vae/config.json",
            "vae/model.safetensors",
        ] {
            #expect(pathSelection.includes(relativePath: requiredPath),
                Comment(stringLiteral: "the Qwen-Image-2.1 offer must download " + requiredPath));
        }
        #expect(pathSelection.includes(relativePath: "qwen-image-2.1.safetensors") == false);
        #expect(pathSelection.includes(relativePath: "editing.jpg") == false);
        #expect(pathSelection.includes(relativePath: "README.md") == true);
    }
    private static func entryJson(
        huggingfaceId: String = "astronomical-test/example-qwen",
        revision: String = "0123456789abcdef0123456789abcdef01234567",
        approximateSizeBytes: UInt64 = 4_000_000_000,
        isPublic: Bool = true,
        extraJson: String = ""
    ) -> String {
        return "{\"huggingface_id\":\"\(huggingfaceId)\",\"revision\":\"\(revision)\","
            + "\"display_name\":\"Example Qwen\",\"family\":\"qwen3_5\","
            + "\"approximate_size_bytes\":\(approximateSizeBytes),\"public\":\(isPublic)\(extraJson)}"
    }

    private static func parseError(_ catalogJson: String) -> DownloadCatalogError? {
        do {
            _ = try DownloadCatalog.parseJson(catalogJson)
            return nil
        } catch let catalogError as DownloadCatalogError {
            return catalogError
        } catch {
            return nil
        }
    }
}
