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
