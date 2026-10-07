import Foundation

import Testing

import AstronomicalConfig
import HuggingFace
import JourneyCategories

@testable import Supervisor

/**
 * Engine seam journeys for the Library download manager (#1059): the real
 * swift-huggingface client runs against the scripted local hub — snapshot
 * files land byte-exact at the destination, live per-file progress is
 * observable mid-transfer, metadata preflight answers gated and private
 * identities, the tree listing exposes digests for verification, and the
 * blob cache serves a second snapshot without re-streaming any payload.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class HubDownloadServiceTests {

    @Test
    func should_download_a_scripted_snapshot_with_live_progress_and_cache_reuse() async throws {
        // 4 MiB at the scripted 16 KiB/40 ms pacing holds the weights
        // transfer open for roughly ten seconds: the upstream progress
        // sampler ticks every 100 ms on the cooperative pool, and under the
        // commit gate's concurrent load a shorter transfer let the whole
        // window pass without one schedulable sample tick.
        let weightsBytes: Data = Data((0 ..< 4 * 1024 * 1024).map({ (byteIndex: Int) -> UInt8 in
            return UInt8(byteIndex % 251)
        }))
        let configBytes: Data = Data("{\"model_family\": \"example\"}".utf8)
        let scriptedHub: ScriptedHuggingFaceHub = try ScriptedHuggingFaceHub(
            repositories: [
            ScriptedHuggingFaceHub.ScriptedRepository(
                repositoryId: "astronomical-test/example-qwen",
                revision: "0123456789abcdef0123456789abcdef01234567",
                gated: false,
                isPrivate: false,
                files: [
                    ScriptedHuggingFaceHub.ScriptedHubFile(
                        relativePath: "config.json",
                        bytes: configBytes,
                        servesAsLfs: false),
                    ScriptedHuggingFaceHub.ScriptedHubFile(
                        relativePath: "weights-00001-of-00001.safetensors",
                        bytes: weightsBytes),
                ]),
        ],
            chunkByteCount: 16 * 1024,
            chunkDelayMilliseconds: 40)
        let hubEndpoint: URL = try await scriptedHub.start()
        defer { scriptedHub.stop(); }

        let temporaryRoot: String = NSTemporaryDirectory() + "ahubdl-\(UUID().uuidString.prefix(8))"
        let cacheDirectory: FilePath = FilePath(string: temporaryRoot + "/hub-cache")
        let destinationDirectory: URL = URL(fileURLWithPath: temporaryRoot + "/models/example-qwen")
        try FileManager.default.createDirectory(atPath: temporaryRoot, withIntermediateDirectories: true)

        let downloadService: HubDownloadService = HubDownloadService(
            hubEndpoint: hubEndpoint,
            cacheDirectory: cacheDirectory)

        let metadata: HubModelMetadata = try await downloadService.fetchModelMetadata(
            repositoryId: "astronomical-test/example-qwen",
            revision: "0123456789abcdef0123456789abcdef01234567")
        #expect(metadata.commitHash == "0123456789abcdef0123456789abcdef01234567")

        let repositoryFiles: Array<HubRepositoryFile> = try await downloadService.listRepositoryFiles(
            repositoryId: "astronomical-test/example-qwen",
            revision: "0123456789abcdef0123456789abcdef01234567")
        #expect(repositoryFiles.count == 2)
        #expect(repositoryFiles[0].sha256Digest == nil, "non-LFS entries carry no independent sha256")

        let liveProgressBox: LiveProgressBox = LiveProgressBox()
        let downloadedDirectory: URL = try await downloadService.downloadSnapshot(
            repositoryId: "astronomical-test/example-qwen",
            revision: "0123456789abcdef0123456789abcdef01234567",
            matching: Array(),
            destinationDirectory: destinationDirectory,
            progressRows: { (fileRows: Array<SnapshotFileProgress>) in
                liveProgressBox.record(fileRows)
            })
        #expect(downloadedDirectory == destinationDirectory)
        let downloadedConfig: Data = try Data(contentsOf: destinationDirectory.appendingPathComponent("config.json"))
        let downloadedWeights: Data = try Data(
            contentsOf: destinationDirectory.appendingPathComponent("weights-00001-of-00001.safetensors"))
        #expect(downloadedConfig == configBytes)
        #expect(downloadedWeights == weightsBytes)
        #expect(
            liveProgressBox.sawIntermediateProgress,
            "paced serving must expose 0 < bytes < total mid-transfer")
        #expect(liveProgressBox.finalCompletedBytes == UInt64(configBytes.count + weightsBytes.count))

        let firstPassStreamedBytes: UInt64 = scriptedHub.streamedPayloadByteCount
        #expect(firstPassStreamedBytes == UInt64(configBytes.count + weightsBytes.count))

        // The second snapshot must come entirely from the blob cache: the hub
        // streams zero new payload bytes for it.
        let repeatDirectory: URL = URL(fileURLWithPath: temporaryRoot + "/models/example-qwen-repeat")
        _ = try await downloadService.downloadSnapshot(
            repositoryId: "astronomical-test/example-qwen",
            revision: "0123456789abcdef0123456789abcdef01234567",
            matching: Array(),
            destinationDirectory: repeatDirectory,
            progressRows: { (_: Array<SnapshotFileProgress>) in
            })
        let repeatWeights: Data = try Data(
            contentsOf: repeatDirectory.appendingPathComponent("weights-00001-of-00001.safetensors"))
        #expect(repeatWeights == weightsBytes)
        #expect(scriptedHub.streamedPayloadByteCount == firstPassStreamedBytes)

        try? FileManager.default.removeItem(atPath: temporaryRoot)
    }

    @Test
    func should_surface_gated_and_private_identities_from_the_metadata_preflight() async throws {
        let scriptedHub: ScriptedHuggingFaceHub = try ScriptedHuggingFaceHub(
            repositories: [
            ScriptedHuggingFaceHub.ScriptedRepository(
                repositoryId: "astronomical-test/gated-model",
                revision: "0123456789abcdef0123456789abcdef01234567",
                gated: true,
                isPrivate: false,
                files: Array()),
            ScriptedHuggingFaceHub.ScriptedRepository(
                repositoryId: "astronomical-test/private-model",
                revision: "0123456789abcdef0123456789abcdef01234567",
                gated: false,
                isPrivate: true,
                files: Array()),
        ])
        let hubEndpoint: URL = try await scriptedHub.start()
        defer { scriptedHub.stop(); }

        let temporaryRoot: String = NSTemporaryDirectory() + "ahubdl-\(UUID().uuidString.prefix(8))"
        try FileManager.default.createDirectory(atPath: temporaryRoot, withIntermediateDirectories: true)
        let downloadService: HubDownloadService = HubDownloadService(
            hubEndpoint: hubEndpoint,
            cacheDirectory: FilePath(string: temporaryRoot + "/hub-cache"))

        await #expect(throws: HubDownloadError.gated) {
            _ = try await downloadService.fetchModelMetadata(
                repositoryId: "astronomical-test/gated-model",
                revision: nil)
        }
        await #expect(throws: HubDownloadError.notPublic) {
            _ = try await downloadService.fetchModelMetadata(
                repositoryId: "astronomical-test/private-model",
                revision: nil)
        }
        try? FileManager.default.removeItem(atPath: temporaryRoot)
    }

    // MARK: - Journey support

    /// Accumulates the sampled per-file rows so the journey can assert both
    /// intermediate and final byte observations.
    private final class LiveProgressBox: @unchecked Sendable {
        private let stateQueue: DispatchQueue = DispatchQueue(label: "live-progress-box")
        private var latestCompletedBytes: UInt64 = 0
        private var observedIntermediateProgress: Bool = false

        func record(_ fileRows: Array<SnapshotFileProgress>) {
            self.stateQueue.sync {
                var completedBytes: UInt64 = 0
                var totalBytes: UInt64 = 0
                for fileRow: SnapshotFileProgress in fileRows {
                    let rowTotal: UInt64 = UInt64(fileRow.sizeBytes ?? 0)
                    totalBytes = totalBytes + rowTotal
                    completedBytes = completedBytes + UInt64((Double(rowTotal) * fileRow.fractionCompleted).rounded())
                }
                if completedBytes > 0 && completedBytes < totalBytes {
                    self.observedIntermediateProgress = true
                }
                self.latestCompletedBytes = completedBytes
            }
        }

        var sawIntermediateProgress: Bool {
            return self.stateQueue.sync(execute: { return self.observedIntermediateProgress; })
        }

        var finalCompletedBytes: UInt64 {
            return self.stateQueue.sync(execute: { return self.latestCompletedBytes; })
        }
    }
}
