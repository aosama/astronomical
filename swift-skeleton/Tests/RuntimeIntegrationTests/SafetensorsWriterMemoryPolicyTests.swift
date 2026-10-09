import Foundation

import MLX
import Testing
import JourneyCategories
import ModelServingTestSupport
import RuntimeIntegration

/**
 * Hermetic SafeTensors writer memory-policy journey, continuing the Rust
 * `safetensors_writer_memory_policy` test: three overlapping strided views
 * of one backing store are written to disk while the MLX wired-memory
 * ceiling sits below the views' combined size, proving the save streams
 * per-array from the source storage instead of materializing every view
 * at once.
 */
extension RuntimeIntegrationMlxJourneyContainer {

    @Suite(.serialized, .tags(.hermeticMlxJourney))
    final class SafetensorsWriterMemoryPolicyTests {

        init() {
            signal(SIGPIPE, SIG_IGN)
            MLXMetallibLocator.overrideMetallibPathIfNecessary()
        }

        @Test(.timeLimit(.minutes(1)))
        func should_write_strided_views_under_a_tight_memory_ceiling() throws {
            let workspaceUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("memory-policy-workspace.safetensors")
            FileManager.default.createFile(atPath: workspaceUrl.path, contents: Data())
            defer {
                try? FileManager.default.removeItem(at: workspaceUrl)
            }

            let memoryLimitGuard: RuntimeMemoryLimitGuard = try Self.writeStreamedViews(to: workspaceUrl)
            defer {
                memoryLimitGuard.restore()
            }

            let workspaceHandle: FileHandle = try FileHandle(forReadingFrom: workspaceUrl)
            defer {
                workspaceHandle.closeFile()
            }
            let weightsFile: SafetensorsFile = try MlxRuntime.loadSafetensors(weightsFile: workspaceHandle)
            #expect(try weightsFile.tensor("largest.weight").shape == [2048, 4094])
            #expect(try weightsFile.tensor("medium.weight").shape == [1536, 4094])
            #expect(try weightsFile.tensor("smallest.weight").shape == [1022, 4094])
        }

        /**
         * Writes the three strided views while the memory policy is pinned by
         * the returned guard; the guard is restored before returning so the
         * read-back in the caller runs under the machine's normal limits.
         */
        private static func writeStreamedViews(to workspaceUrl: URL) throws -> RuntimeMemoryLimitGuard {
            let backingStore: MLXArray = MLX.zeros([4096, 4096], dtype: .bfloat16)
            MLX.eval([backingStore])

            let largestView: MLXArray = MLX.asStrided(backingStore, [2048, 4094], strides: [4096, 1], offset: 4097)
            let mediumView: MLXArray = MLX.asStrided(backingStore, [1536, 4094], strides: [4096, 1], offset: 2049 * 4096 + 1)
            let smallestView: MLXArray = MLX.asStrided(backingStore, [1022, 4094], strides: [4096, 1], offset: 3073 * 4096 + 1)
            #expect(largestView.nbytes > mediumView.nbytes)
            #expect(mediumView.nbytes > smallestView.nbytes)

            let baselineActiveMemoryBytes: Int = MLX.Memory.activeMemory
            let memoryCeilingBytes: Int = baselineActiveMemoryBytes + largestView.nbytes + 2_000_000
            // The ceiling must sit below the views' combined size, otherwise
            // the journey would prove nothing about streaming.
            #expect(largestView.nbytes + mediumView.nbytes > memoryCeilingBytes - baselineActiveMemoryBytes)

            let memoryLimitGuard: RuntimeMemoryLimitGuard = RuntimeMemoryLimitGuard(activeMemoryLimitBytes: memoryCeilingBytes)
            let writeOutcome: SafetensorsWriteOutcome = try MlxRuntime.saveSafetensors(
                arrays: [
                    "largest.weight": largestView,
                    "medium.weight": mediumView,
                    "smallest.weight": smallestView,
                ],
                metadata: ["format_version": "streaming-workspace-test"],
                to: workspaceUrl)
            #expect(writeOutcome.writtenByteCount > 0)

            memoryLimitGuard.restore()
            return memoryLimitGuard
        }
    }
}

/**
 * Pins the MLX memory policy for the duration of one streaming write and
 * restores the machine's original limits afterwards. Restore is idempotent
 * and also runs from deinit, so the pinned policy can never leak past the
 * test even on a throwing path.
 */
private final class RuntimeMemoryLimitGuard {

    private let stateLock: NSLock = NSLock()
    private let originalCacheLimitBytes: Int
    private let originalMemoryLimitBytes: Int
    private var didRestoreLimits: Bool = false

    init(activeMemoryLimitBytes: Int) {
        self.originalCacheLimitBytes = MLX.Memory.cacheLimit
        self.originalMemoryLimitBytes = MLX.Memory.memoryLimit
        MLX.Memory.cacheLimit = 0
        MLX.Memory.memoryLimit = activeMemoryLimitBytes
    }

    func restore() {
        stateLock.lock()
        defer {
            stateLock.unlock()
        }
        guard (!didRestoreLimits) else {
            return
        }
        didRestoreLimits = true
        MLX.Memory.cacheLimit = originalCacheLimitBytes
        MLX.Memory.memoryLimit = originalMemoryLimitBytes
        MLX.Memory.clearCache()
    }

    deinit {
        restore()
    }
}
