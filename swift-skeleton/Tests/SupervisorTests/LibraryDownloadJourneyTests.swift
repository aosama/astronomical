import Foundation

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;
import JourneyCategories;

@testable import Supervisor;

/**
 * Library download journeys against a scripted local hub: a POST download
 * request is accepted, transfers, verifies, and publishes into the catalog
 * readiness join; a failing discovery refresh strands the job in
 * publishing; and the single-slot rule rejects a second concurrent
 * download. Mirrors rest_api/library_download.rs per the #1059 shape.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class LibraryDownloadJourneyTests {

    static let catalogJson: String = """
        {
            "schema_version": 2,
            "entries": [
                {
                    "huggingface_id": "astronomical-test/example-qwen",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "display_name": "Example Qwen",
                    "family": "qwen3_5",
                    "approximate_size_bytes": 400000,
                    "public": true
                }
            ]
        }
        """;

    @Test
    func should_accept_a_download_request_and_publish_the_model_into_the_catalog() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }

        let startResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let (startStatusCode, startEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(startResponseText)
        #expect(startStatusCode == 202)
        #expect(startEnvelope["state"] as? String == "checking_disk")
        #expect(startEnvelope["huggingface_id"] as? String == "astronomical-test/example-qwen")

        let readyEntry: [String: Any] = try journeyHarness.pollUntilCatalogEntryReady()
        #expect(readyEntry["requestable_model_id"] as? String == "example-qwen")

        let publishedConfig: Data = try Data(
            contentsOf: URL(fileURLWithPath: journeyHarness.modelsDirectory
                .appending(component: "astronomical-test")
                .appending(component: "example-qwen")
                .appending(component: "config.json").string))
        #expect(publishedConfig == Data("{\"model_family\": \"example\"}".utf8))
        let provenancePath: FilePath = journeyHarness.modelsDirectory
            .appending(component: "astronomical-test")
            .appending(component: "example-qwen")
            .appending(component: ".astronomical-library-provenance.json")
        #expect(FileManager.default.fileExists(atPath: provenancePath.string))

        let idleResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "GET /v1/library/download HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (idleStatusCode, idleEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(idleResponseText)
        #expect(idleStatusCode == 200)
        #expect(idleEnvelope["state"] as? String == "idle")
    }

    @Test
    func should_strand_the_job_in_publishing_when_the_discovery_refresh_fails() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .fails)
        defer { journeyHarness.stop() }

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())

        let strandedEnvelope: [String: Any] = try journeyHarness.pollUntilDownloadState("publishing")
        #expect(strandedEnvelope["bytes_total"] as? UInt64 == UInt64(journeyHarness.expectedTotalBytes))

        let (catalogStatusCode, catalogEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(
            RawLoopbackHttpClient.exchange(
                port: journeyHarness.server.boundEndpoint.port,
                requestText: "GET /v1/library/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"))
        #expect(catalogStatusCode == 200)
        let catalogEntries: [[String: Any]] = catalogEnvelope["entries"] as? [[String: Any]] ?? []
        #expect(catalogEntries.count == 1)
        #expect(catalogEntries[0]["ready_on_this_mac"] as? Bool == false)
        #expect(catalogEntries[0]["download_state"] as? String == "publishing")
    }

    @Test
    func should_reject_a_second_download_while_one_is_active() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        _ = try journeyHarness.pollUntilAnyDownloadState()

        let secondResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let (secondStatusCode, secondEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(secondResponseText)
        #expect(secondStatusCode == 409)
        let errorObject: [String: Any] = secondEnvelope["error"] as? [String: Any] ?? [:]
        #expect(errorObject["code"] as? String == "library_busy")

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "POST /v1/library/download/cancel HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    }

    @Test
    func should_pause_a_download_and_resume_it_across_a_coordinator_restart() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds,
            weightsByteCount: 512 * 1024,
            pacing: LibraryDownloadJourneyHarness.TransferPacing(
                chunkByteCount: 16 * 1024,
                chunkDelayMilliseconds: 25))
        defer { journeyHarness.stop() }

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        _ = try journeyHarness.pollUntilLiveProgressObserved()

        let pauseResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "POST /v1/library/download/pause HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (pauseStatusCode, pauseEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(pauseResponseText)
        #expect(pauseStatusCode == 200)
        #expect(pauseEnvelope["state"] as? String == "paused")

        try await journeyHarness.restart()
        let resumeResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "POST /v1/library/download/resume HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (resumeStatusCode, resumeEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(resumeResponseText)
        #expect(resumeStatusCode == 202)
        #expect(resumeEnvelope["state"] as? String == "resuming")

        let readyEntry: [String: Any] = try journeyHarness.pollUntilCatalogEntryReady()
        #expect(readyEntry["huggingface_id"] as? String == "astronomical-test/example-qwen")
    }

    @Test
    func should_report_live_progress_without_durable_write_amplification_and_cancel() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds,
            weightsByteCount: 512 * 1024,
            pacing: LibraryDownloadJourneyHarness.TransferPacing(
                chunkByteCount: 16 * 1024,
                chunkDelayMilliseconds: 25))
        defer { journeyHarness.stop() }

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let liveEnvelope: [String: Any] = try journeyHarness.pollUntilLiveProgressObserved()
        let liveCompletedBytes: UInt64 = liveEnvelope["bytes_completed"] as? UInt64 ?? 0
        #expect(liveCompletedBytes > 0)
        #expect(liveCompletedBytes < (liveEnvelope["bytes_total"] as? UInt64 ?? 0))

        let durableRecord: LibraryDownloadJobRecord? = journeyHarness.durableRecordOnDisk()
        #expect(durableRecord?.bytesCompleted == 0, "live progress must never amplify durable writes")

        let cancelResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "POST /v1/library/download/cancel HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (cancelStatusCode, cancelEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(cancelResponseText)
        #expect(cancelStatusCode == 200)
        #expect(cancelEnvelope["state"] as? String == "idle")
        #expect(journeyHarness.hasIncompleteCacheBlobs() == false, "cancel must purge partial blobs")
        #expect(
            FileManager.default.fileExists(
                atPath: journeyHarness.modelsDirectory
                    .appending(component: "astronomical-test").string) == false,
            "no partial model directory survives a cancel")
        #expect(journeyHarness.durableRecordOnDisk() == nil, "cancel deletes the durable job")
    }

    @Test
    func should_surface_gated_and_checksum_failures_as_stable_error_codes() async throws {
        let gatedHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds,
            gated: true)
        defer { gatedHarness.stop() }
        _ = RawLoopbackHttpClient.exchange(
            port: gatedHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let gatedEnvelope: [String: Any] = try gatedHarness.pollUntilDownloadState("failed")
        #expect(gatedEnvelope["error_code"] as? String == "download_gated")
        #expect(
            FileManager.default.fileExists(
                atPath: gatedHarness.modelsDirectory
                    .appending(component: "astronomical-test").string) == false)

        let corruptHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { corruptHarness.stop() }
        corruptHarness.scriptedHub.corruptServedBytes(
            repositoryId: "astronomical-test/example-qwen",
            relativePath: "weights-00001-of-00001.safetensors",
            bytes: Data(repeating: 0x63, count: 256 * 1024))
        _ = RawLoopbackHttpClient.exchange(
            port: corruptHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let checksumEnvelope: [String: Any] = try corruptHarness.pollUntilDownloadState("failed")
        #expect(checksumEnvelope["error_code"] as? String == "checksum_mismatch")
        #expect(
            FileManager.default.fileExists(
                atPath: corruptHarness.modelsDirectory
                    .appending(component: "astronomical-test").string) == false,
            "a corrupt transfer must never appear under the models directory")
    }

    @Test
    func should_redownload_and_succeed_after_a_checksum_failure() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }
        journeyHarness.scriptedHub.corruptServedBytes(
            repositoryId: "astronomical-test/example-qwen",
            relativePath: "weights-00001-of-00001.safetensors",
            bytes: Data(repeating: 0x63, count: 256 * 1024))
        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        _ = try journeyHarness.pollUntilDownloadState("failed")

        journeyHarness.scriptedHub.clearServedByteCorruption(
            repositoryId: "astronomical-test/example-qwen",
            relativePath: "weights-00001-of-00001.safetensors")
        let resumeResponseText: String? = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: "POST /v1/library/download/resume HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        let (resumeStatusCode, _): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(resumeResponseText)
        #expect(resumeStatusCode == 202)

        let readyEntry: [String: Any] = try journeyHarness.pollUntilCatalogEntryReady()
        #expect(readyEntry["ready_on_this_mac"] as? Bool == true)
        #expect(journeyHarness.scriptedHub.streamedPayloadByteCount > journeyHarness.expectedTotalBytes,
            "the retry must have redownloaded the purged payload")
    }

    @Test
    func should_adopt_an_exact_pre_existing_publication_after_manifest_verification() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }
        let publishedDirectory: FilePath = journeyHarness.modelsDirectory
            .appending(component: "astronomical-test")
            .appending(component: "example-qwen")
        try FileManager.default.createDirectory(
            atPath: publishedDirectory.string,
            withIntermediateDirectories: true)
        try journeyHarness.scriptedConfigBytes.write(
            to: URL(fileURLWithPath: publishedDirectory.appending(component: "config.json").string))
        try journeyHarness.scriptedWeightsBytes.write(
            to: URL(fileURLWithPath: publishedDirectory.appending(component: "weights-00001-of-00001.safetensors").string))

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        _ = try journeyHarness.pollUntilDownloadState("idle")
        let readyEntry: [String: Any] = try journeyHarness.pollUntilCatalogEntryReady()
        #expect(readyEntry["ready_on_this_mac"] as? Bool == true)
        let provenancePath: FilePath = publishedDirectory.appending(
            component: ".astronomical-library-provenance.json")
        #expect(FileManager.default.fileExists(atPath: provenancePath.string))
    }

    @Test
    func should_leave_a_mismatched_pre_existing_publication_untouched() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }
        let publishedDirectory: FilePath = journeyHarness.modelsDirectory
            .appending(component: "astronomical-test")
            .appending(component: "example-qwen")
        try FileManager.default.createDirectory(
            atPath: publishedDirectory.string,
            withIntermediateDirectories: true)
        let mismatchedConfig: Data = Data("{\"model_family\": \"other-model\"}".utf8)
        try mismatchedConfig.write(
            to: URL(fileURLWithPath: publishedDirectory.appending(component: "config.json").string))

        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        let failedEnvelope: [String: Any] = try journeyHarness.pollUntilDownloadState("failed")
        #expect(failedEnvelope["error_code"] as? String == "model_already_present")
        let survivingConfig: Data = try Data(
            contentsOf: URL(fileURLWithPath: publishedDirectory.appending(component: "config.json").string))
        #expect(survivingConfig == mismatchedConfig, "a mismatched destination is never rewritten")
        #expect(
            FileManager.default.fileExists(
                atPath: publishedDirectory.appending(
                    component: ".astronomical-library-provenance.json").string) == false,
            "a mismatched destination must not receive Library provenance")
    }

    @Test
    func should_remain_ready_after_the_coordinator_restarts() async throws {
        let journeyHarness: LibraryDownloadJourneyHarness = try await LibraryDownloadJourneyHarness(
            discoveryRefreshOutcome: .succeeds)
        defer { journeyHarness.stop() }
        _ = RawLoopbackHttpClient.exchange(
            port: journeyHarness.server.boundEndpoint.port,
            requestText: LibraryDownloadJourneyTests.startDownloadRequestText())
        _ = try journeyHarness.pollUntilDownloadState("idle")
        _ = try journeyHarness.pollUntilCatalogEntryReady()

        try await journeyHarness.restart()
        let (statusCode, catalogEnvelope): (Int, [String: Any]) = try LibraryDownloadJourneyTests.decodeEnvelope(
            RawLoopbackHttpClient.exchange(
                port: journeyHarness.server.boundEndpoint.port,
                requestText: "GET /v1/library/catalog HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"))
        #expect(statusCode == 200)
        let catalogEntries: [[String: Any]] = catalogEnvelope["entries"] as? [[String: Any]] ?? []
        #expect(catalogEntries.count == 1)
        #expect(catalogEntries[0]["ready_on_this_mac"] as? Bool == true)
        #expect(catalogEntries[0]["download_state"] == nil || catalogEntries[0]["download_state"] as? String == nil)
        #expect(catalogEntries[0]["requestable_model_id"] as? String == "example-qwen")
    }

    // MARK: - Journey support

    private static func startDownloadRequestText() -> String {
        let requestBody: String = "{\"huggingface_id\": \"astronomical-test/example-qwen\"}"
        return "POST /v1/library/download HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            + "Content-Type: application/json\r\nContent-Length: \(requestBody.utf8.count)\r\n\r\n"
            + requestBody
    }

    private static func decodeEnvelope(_ responseText: String?) throws -> (Int, [String: Any]) {
        guard let unwrappedResponseText: String = responseText else {
            throw LibraryDownloadJourneyFailure.missingResponse
        }
        guard let statusToken: Substring = unwrappedResponseText.split(separator: " ", maxSplits: 2).dropFirst().first,
            let statusCode: Int = Int(statusToken)
        else {
            throw LibraryDownloadJourneyFailure.malformedStatusLine
        }
        guard let bodyStart: String.Index = unwrappedResponseText.range(of: "\r\n\r\n")?.upperBound else {
            throw LibraryDownloadJourneyFailure.missingResponseBody
        }
        let bodyJsonText: String = String(unwrappedResponseText[bodyStart...])
        guard let envelope: [String: Any] = try? JSONSerialization.jsonObject(with: Data(bodyJsonText.utf8)) as? [String: Any] else {
            throw LibraryDownloadJourneyFailure.nonJsonEnvelope
        }
        return (statusCode, envelope)
    }
}
