import Foundation

import Testing

@testable import Supervisor

/**
 * Hermetic coverage for the supervisor performance attribution log,
 * migrating the journeys of
 * apps/supervisor/tests/hermetic/supervisor_performance_attribution.rs: the
 * disabled log never reads the clock, enabled measurements append one
 * deterministic JSON row with flattened download evidence, best-effort
 * failures never alter the operation output, the writer lock is free while a
 * measured async operation awaits, and file-transfer attribution accepts
 * only canonical relative paths.
 */
@Suite(.tags(.hermeticJourney))
final class SupervisorPerformanceAttributionTests {

    @Test
    func should_execute_disabled_async_measurement_without_reading_the_clock() async throws -> Void {
        let clockCounter: JourneyClockCounter = JourneyClockCounter()
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClockWhenEnabled(
                writer: DiscardAttributionSink(),
                unixEpochMillis: { () throws -> UInt64 in
                    return clockCounter.nextValue()
                },
                performanceAttributionEnabled: false)

        let operationOutput: UInt64 = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> UInt64 in
                return try await attributionLog.measureAsyncOperation(
                    operation: .manifestFetch,
                    measuredOperation: { () async -> UInt64 in
                        return 42
                    },
                    describeMeasurement: { (_: UInt64) -> SupervisorPerformanceMeasurement in
                        return SupervisorPerformanceMeasurement.success()
                    })
            })
        #expect(operationOutput == 42)
        #expect(clockCounter.totalCalls == 0)
    }

    @Test
    func should_record_deterministic_async_manifest_measurement() async throws -> Void {
        let sharedSink: SharedMemoryAttributionSink = SharedMemoryAttributionSink()
        let sequencedClock: JourneySequencedClock = JourneySequencedClock(values: [1_000, 1_025])
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: sharedSink,
                unixEpochMillis: { () throws -> UInt64 in
                    return try sequencedClock.nextValue()
                })
        let manifestMeasurement: SupervisorPerformanceMeasurement = try SupervisorPerformanceMeasurement
            .success()
            .withManifestFetch(
                huggingfaceId: "astronomical-test/example-qwen",
                revision: AttributionJourneySupport.TEST_REVISION,
                manifestFileCount: 3,
                manifestTotalBytes: 4_000_000_000)

        let operationOutput: String = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> String in
                return try await attributionLog.measureAsyncOperation(
                    operation: .manifestFetch,
                    measuredOperation: { () async -> String in
                        return "manifest fetched"
                    },
                    describeMeasurement: { (_: String) -> SupervisorPerformanceMeasurement in
                        return manifestMeasurement
                    })
            })

        #expect(operationOutput == "manifest fetched")
        let attributionRecord: [String: Any] = try SupervisorPerformanceAttributionTests.parseOnlyRecord(
            sharedSink)
        #expect(attributionRecord["operation"] as? String == "manifest_fetch")
        #expect(attributionRecord["started_at_unix_millis"] as? Int == 1_000)
        #expect(attributionRecord["ended_at_unix_millis"] as? Int == 1_025)
        #expect(attributionRecord["outcome"] as? String == "success")
        #expect(attributionRecord["huggingface_id"] as? String == "astronomical-test/example-qwen")
        #expect(attributionRecord["revision"] as? String == AttributionJourneySupport.TEST_REVISION)
        #expect(attributionRecord["manifest_file_count"] as? Int == 3)
        #expect(attributionRecord["manifest_total_bytes"] as? Int == 4_000_000_000)
    }

    @Test
    func should_record_qwen_thinking_seed_loading_without_download_metadata() async throws -> Void {
        let sharedSink: SharedMemoryAttributionSink = SharedMemoryAttributionSink()
        let sequencedClock: JourneySequencedClock = JourneySequencedClock(values: [2_000, 2_005])
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: sharedSink,
                unixEpochMillis: { () throws -> UInt64 in
                    return try sequencedClock.nextValue()
                })

        _ = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> Void in
                return try await attributionLog.measureAsyncOperation(
                    operation: .libraryCatalogLoad,
                    measuredOperation: { () async -> Void in
                        return
                    },
                    describeMeasurement: { (_: Void) -> SupervisorPerformanceMeasurement in
                        return SupervisorPerformanceMeasurement.success()
                    })
            })

        let attributionRecord: [String: Any] = try SupervisorPerformanceAttributionTests.parseOnlyRecord(
            sharedSink)
        #expect(attributionRecord["operation"] as? String == "library_catalog_load")
        #expect(attributionRecord["started_at_unix_millis"] as? Int == 2_000)
        #expect(attributionRecord["ended_at_unix_millis"] as? Int == 2_005)
        #expect(attributionRecord["outcome"] as? String == "success")
    }

    @Test
    func should_preserve_best_effort_operation_output_when_the_start_clock_fails() async throws -> Void {
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: DiscardAttributionSink(),
                unixEpochMillis: { () throws -> UInt64 in
                    throw IntentionalStartClockFailure()
                })

        let operationOutput: String = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> String in
                return await attributionLog.measureAsyncOperationBestEffort(
                    operation: .libraryCatalogLoad,
                    measuredOperation: { () async -> String in
                        return "Romeo and Juliet"
                    },
                    describeMeasurement: { (_: String) -> SupervisorPerformanceMeasurement in
                        return SupervisorPerformanceMeasurement.success()
                    })
            })
        #expect(operationOutput == "Romeo and Juliet")
    }

    @Test
    func should_preserve_best_effort_operation_output_when_the_writer_fails() async throws -> Void {
        let operationCompleted: JourneyFlag = JourneyFlag()
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: CompletionCheckingAttributionSink(operationCompleted: { () -> Bool in
                    return operationCompleted.isSet
                }),
                unixEpochMillis: { () throws -> UInt64 in
                    return 1_000
                })

        let operationOutput: String = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> String in
                return await attributionLog.measureAsyncOperationBestEffort(
                    operation: .libraryCatalogLoad,
                    measuredOperation: { () async -> String in
                        operationCompleted.set()
                        return "Two households"
                    },
                    describeMeasurement: { (_: String) -> SupervisorPerformanceMeasurement in
                        return SupervisorPerformanceMeasurement.success()
                    })
            })
        #expect(operationOutput == "Two households")
    }

    @Test
    func should_not_hold_the_writer_lock_while_an_async_operation_is_blocked() async throws -> Void {
        let sharedSink: SharedMemoryAttributionSink = SharedMemoryAttributionSink()
        let clockCounter: JourneyClockCounter = JourneyClockCounter()
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: sharedSink,
                unixEpochMillis: { () throws -> UInt64 in
                    return clockCounter.nextValue()
                })
        let firstOperationStarted: JourneyGate = JourneyGate()
        let releaseFirstOperation: JourneyGate = JourneyGate()

        let firstOperation: Task<Void, any Error> = Task<Void, any Error> { () throws -> Void in
            _ = try await attributionLog.measureAsyncOperation(
                operation: .fileTransfer,
                measuredOperation: { () async -> Void in
                    await firstOperationStarted.signal()
                    await releaseFirstOperation.waitUntilSignaled()
                },
                describeMeasurement: { (_: Void) -> SupervisorPerformanceMeasurement in
                    return SupervisorPerformanceAttributionTests.fileTransferMeasurement()
                })
        }
        try await AttributionJourneySupport.awaitBounded({ () async throws -> Void in
            await firstOperationStarted.waitUntilSignaled()
        })

        _ = try await attributionLog.measureAsyncOperation(
            operation: .verification,
            measuredOperation: { () async -> Void in
                return
            },
            describeMeasurement: { (_: Void) -> SupervisorPerformanceMeasurement in
                return SupervisorPerformanceAttributionTests.verificationMeasurement()
            })
        #expect(sharedSink.recordCount() == 1)

        await releaseFirstOperation.signal()
        try await AttributionJourneySupport.awaitBounded({ () async throws -> Void in
            try await firstOperation.value
        })
        #expect(sharedSink.recordCount() == 2)
    }

    @Test
    func should_report_end_clock_failure_only_after_async_operation_completes() async throws -> Void {
        let operationCompleted: JourneyFlag = JourneyFlag()
        let clockCounter: JourneyClockCounter = JourneyClockCounter()
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: DiscardAttributionSink(),
                unixEpochMillis: { () throws -> UInt64 in
                    let clockValue: UInt64 = clockCounter.nextValue()
                    if clockCounter.totalCalls == 1 {
                        return clockValue
                    }
                    guard operationCompleted.isSet else {
                        throw JourneyAssertionFailure(
                            problem: "the end clock must only be read after the operation completes")
                    }
                    throw IntentionalEndClockFailure()
                })

        let attributionFailure: JourneyErrorCapture = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> JourneyErrorCapture in
                do {
                    _ = try await attributionLog.measureAsyncOperation(
                        operation: .diskPreflight,
                        measuredOperation: { () async -> Void in
                            operationCompleted.set()
                        },
                        describeMeasurement: { (_: Void) -> SupervisorPerformanceMeasurement in
                            return SupervisorPerformanceAttributionTests.diskPreflightMeasurement()
                        })
                    throw JourneyAssertionFailure(problem: "the end clock failure must surface as a typed error")
                } catch let attributionError {
                    return JourneyErrorCapture(failureDescription: String(describing: attributionError))
                }
            })
        #expect(attributionFailure.failureDescription.contains("intentional end clock failure"))
    }

    @Test
    func should_report_write_failure_only_after_async_operation_completes() async throws -> Void {
        let operationCompleted: JourneyFlag = JourneyFlag()
        let attributionLog: SupervisorPerformanceAttributionLog =
            SupervisorPerformanceAttributionLog.fromWriterAndClock(
                writer: CompletionCheckingAttributionSink(operationCompleted: { () -> Bool in
                    return operationCompleted.isSet
                }),
                unixEpochMillis: { () throws -> UInt64 in
                    return 1_000
                })

        let attributionFailure: JourneyErrorCapture = try await AttributionJourneySupport.awaitBounded(
            { () async throws -> JourneyErrorCapture in
                do {
                    _ = try await attributionLog.measureAsyncOperation(
                        operation: .verification,
                        measuredOperation: { () async -> Void in
                            operationCompleted.set()
                        },
                        describeMeasurement: { (_: Void) -> SupervisorPerformanceMeasurement in
                            return SupervisorPerformanceAttributionTests.verificationMeasurement()
                        })
                    throw JourneyAssertionFailure(problem: "the write failure must surface as a typed error")
                } catch let attributionError {
                    return JourneyErrorCapture(failureDescription: String(describing: attributionError))
                }
            })
        #expect(attributionFailure.failureDescription.contains("intentional write failure"))
    }

    @Test
    func should_reject_noncanonical_file_transfer_attribution_paths() throws -> Void {
        let noncanonicalRelativePaths: Array<String> = [
            "weights//model.safetensors",
            "weights/./model.safetensors",
            "weights/model.safetensors/",
        ]
        for noncanonicalRelativePath: String in noncanonicalRelativePaths {
            #expect(
                SupervisorPerformanceAttributionTests.rejectsFileTransferPath(noncanonicalRelativePath),
                "attribution must use the canonical durable file identity, rejected path: \(noncanonicalRelativePath)")
        }
    }

    // MARK: Journey fixtures

    private static func rejectsFileTransferPath(_ relativeFilePath: String) -> Bool {
        do {
            _ = try SupervisorPerformanceMeasurement.success().withFileTransfer(
                huggingfaceId: "astronomical-test/example-qwen",
                revision: AttributionJourneySupport.TEST_REVISION,
                relativeFilePath: relativeFilePath,
                resumeOffsetBytes: 0,
                transferredBytes: 1)
            return false
        } catch SupervisorPerformanceAttributionError.invalidRelativeFilePath {
            return true
        } catch {
            return false
        }
    }

    private static func diskPreflightMeasurement() -> SupervisorPerformanceMeasurement {
        var measurement: SupervisorPerformanceMeasurement = SupervisorPerformanceMeasurement.failure()
        measurement.downloadDetail = SupervisorDownloadMeasurementDetail.validated(
            huggingfaceId: "astronomical-test/example-qwen",
            revision: AttributionJourneySupport.TEST_REVISION,
            operationDetail: .diskPreflight(requiredBytes: 100, availableBytes: 50))
        return measurement
    }

    private static func fileTransferMeasurement() -> SupervisorPerformanceMeasurement {
        var measurement: SupervisorPerformanceMeasurement = SupervisorPerformanceMeasurement.success()
        measurement.downloadDetail = SupervisorDownloadMeasurementDetail.validated(
            huggingfaceId: "astronomical-test/example-qwen",
            revision: AttributionJourneySupport.TEST_REVISION,
            operationDetail: .fileTransfer(
                relativeFilePath: "model.safetensors",
                resumeOffsetBytes: 0,
                transferredBytes: 100))
        return measurement
    }

    private static func verificationMeasurement() -> SupervisorPerformanceMeasurement {
        var measurement: SupervisorPerformanceMeasurement = SupervisorPerformanceMeasurement.cancelled()
        measurement.downloadDetail = SupervisorDownloadMeasurementDetail.validated(
            huggingfaceId: "astronomical-test/example-qwen",
            revision: AttributionJourneySupport.TEST_REVISION,
            operationDetail: .verification(verifiedFileCount: 1, verifiedBytes: 100))
        return measurement
    }

    private static func parseOnlyRecord(_ sharedSink: SharedMemoryAttributionSink) throws -> [String: Any] {
        let writtenBytes: Data = sharedSink.writtenBytes()
        let parsedDocument: Any = try JSONSerialization.jsonObject(with: writtenBytes, options: [])
        guard let parsedObject: [String: Any] = parsedDocument as? [String: Any] else {
            throw JourneyAssertionFailure(problem: "exactly one JSON record must be written")
        }
        return parsedObject
    }
}

struct IntentionalStartClockFailure: Error, CustomStringConvertible {
    var description: String {
        return "intentional start clock failure"
    }
}

struct IntentionalEndClockFailure: Error, CustomStringConvertible {
    var description: String {
        return "intentional end clock failure"
    }
}

/// Carries a captured failure text across the bounded await boundary.
struct JourneyErrorCapture: Error, Sendable {
    let failureDescription: String
}
