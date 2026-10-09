import Foundation

import Testing
import JourneyCategories
import ModelServingTestSupport
import RuntimeIntegration

/**
 * Hermetic bounded-interval validation journeys: every way a set of
 * bounded read intervals can fail to tile its virtual payload is refused
 * by the public loader before a single byte is read. The Rust tree had no
 * refusal coverage for this validator, so these journeys are new guard
 * surface at the Swift boundary rather than a direct port.
 */
extension RuntimeIntegrationMlxJourneyContainer {

    @Suite(.serialized, .tags(.hermeticMlxJourney))
    final class BoundedIntervalValidationTests {

        init() {
            signal(SIGPIPE, SIG_IGN)
            MLXMetallibLocator.overrideMetallibPathIfNecessary()
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_an_empty_interval_set() throws {
            do {
                try Self.attemptBoundedLoad(intervals: [], totalPayloadBytes: 8)
                Issue.record("an empty interval set must fail validation")
            } catch MlxRuntimeError.boundedIntervalValidation(let description) {
                #expect(!description.isEmpty)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_a_chain_that_skips_the_payload_origin() throws {
            do {
                try Self.attemptBoundedLoad(
                    intervals: [BoundedReadInterval(virtualPayloadOffset: 4, sourceFileOffset: 12, sourceByteCount: 4)],
                    totalPayloadBytes: 8)
                Issue.record("a chain that does not start at the payload origin must fail validation")
            } catch MlxRuntimeError.boundedIntervalValidation(let description) {
                #expect(!description.isEmpty)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_a_chain_with_a_virtual_payload_gap() throws {
            let gappedIntervals: [BoundedReadInterval] = [
                BoundedReadInterval(virtualPayloadOffset: 0, sourceFileOffset: 12, sourceByteCount: 4),
                BoundedReadInterval(virtualPayloadOffset: 8, sourceFileOffset: 16, sourceByteCount: 8),
            ]
            do {
                try Self.attemptBoundedLoad(intervals: gappedIntervals, totalPayloadBytes: 16)
                Issue.record("a chain with a gap between intervals must fail validation")
            } catch MlxRuntimeError.boundedIntervalValidation(let description) {
                #expect(!description.isEmpty)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_a_chain_shorter_than_the_declared_payload() throws {
            let shortChainIntervals: [BoundedReadInterval] = [
                BoundedReadInterval(virtualPayloadOffset: 0, sourceFileOffset: 12, sourceByteCount: 4),
                BoundedReadInterval(virtualPayloadOffset: 4, sourceFileOffset: 16, sourceByteCount: 4),
            ]
            do {
                try Self.attemptBoundedLoad(intervals: shortChainIntervals, totalPayloadBytes: 16)
                Issue.record("a chain that ends short of the declared payload must fail validation")
            } catch MlxRuntimeError.boundedIntervalValidation(let description) {
                #expect(!description.isEmpty)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        @Test(.timeLimit(.minutes(1)))
        func should_refuse_source_ranges_that_overlap_in_the_weights_file() throws {
            let overlappingSourceIntervals: [BoundedReadInterval] = [
                BoundedReadInterval(virtualPayloadOffset: 0, sourceFileOffset: 12, sourceByteCount: 8),
                BoundedReadInterval(virtualPayloadOffset: 8, sourceFileOffset: 16, sourceByteCount: 8),
            ]
            do {
                try Self.attemptBoundedLoad(intervals: overlappingSourceIntervals, totalPayloadBytes: 16)
                Issue.record("source ranges that overlap inside the weights file must fail validation")
            } catch MlxRuntimeError.boundedIntervalValidation(let description) {
                #expect(!description.isEmpty)
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }

        /**
         * Drives the public loader against a tiny on-disk fixture; the
         * validator runs before any read, so refusal journeys never touch
         * the payload bytes and the fixture only needs to exist as a real
         * file with an open descriptor.
         */
        private static func attemptBoundedLoad(
            intervals: [BoundedReadInterval],
            totalPayloadBytes: UInt64
        ) throws {
            let headerJson: String = "{\"probe.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}"
            let headerJsonBytes: [UInt8] = Array(headerJson.utf8)
            var syntheticHeaderBytes: [UInt8] = SafetensorsFixtureSupport.littleEndianLengthPrefix(of: UInt64(headerJsonBytes.count))
            syntheticHeaderBytes.append(contentsOf: headerJsonBytes)

            var fixtureBytes: [UInt8] = syntheticHeaderBytes
            fixtureBytes.append(contentsOf: SafetensorsFixtureSupport.littleEndianBytes(of: [1, 2]))
            let weightsFileUrl: URL = SafetensorsFixtureSupport.temporaryFileUrl("bounded-validation.safetensors")
            FileManager.default.createFile(atPath: weightsFileUrl.path, contents: Data(fixtureBytes))
            defer {
                try? FileManager.default.removeItem(at: weightsFileUrl)
            }

            let sourceFileHandle: FileHandle = try FileHandle(forReadingFrom: weightsFileUrl)
            defer {
                sourceFileHandle.closeFile()
            }

            _ = try MlxRuntime.loadSafetensorsFromBoundedRanges(
                sourceFile: sourceFileHandle,
                syntheticHeaderBytes: Data(syntheticHeaderBytes),
                intervals: intervals,
                totalPayloadBytes: totalPayloadBytes)
        }
    }
}
