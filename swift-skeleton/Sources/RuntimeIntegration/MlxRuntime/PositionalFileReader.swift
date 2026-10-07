import Foundation

/**
 * Positional (pread-based) file reading for the SafeTensors I/O surface,
 * continuing the Rust `PositionalFileReader` contract: gather many source
 * ranges of one open file descriptor into a single assembled buffer with a
 * bounded number of concurrent reads, so SSD (Solid State Drive) streaming
 * parallelism stays observable through `PositionalFileReadMetrics`.
 *
 * The concurrency ceiling adapts to the machine
 * (`DEFAULT_MAXIMUM_CONCURRENT_READ_COUNT` derives from the active
 * processor count); nothing here is tuned to one workstation.
 */
enum PositionalFileReader {

    /** One source range to place at a chosen offset of the assembled buffer. */
    struct ReadRequest {

        let sourceFileOffset: UInt64
        let destinationBufferOffset: Int
        let byteCount: Int
    }

    /** A `ReadRequest` piece already bounded to at most `CHUNK_BYTE_COUNT`. */
    struct ReadChunk: Sendable {

        let sourceFileOffset: UInt64
        let destinationBufferOffset: Int
        let byteCount: Int
    }

    /// The largest single pread a worker issues; larger requests are split
    /// into sequential pieces so one huge tensor cannot monopolize a permit.
    static let CHUNK_BYTE_COUNT: Int = 4 * 1024 * 1024

    /// Cache-line alignment for the assembled buffer; MLX consumers read the
    /// bytes as tensor payloads, so aligned starts avoid split cache lines.
    static let ASSEMBLED_BUFFER_ALIGNMENT: Int = 64

    /// The machine-adaptive default for how many chunk reads may overlap in
    /// time: never more than four, never more than the active processors.
    static let DEFAULT_MAXIMUM_CONCURRENT_READ_COUNT: Int =
        min(4, ProcessInfo.processInfo.activeProcessorCount)

    /**
     * - Returns: the byte size of the open file behind `fileDescriptor`.
     * - Throws: `MlxRuntimeError.positionalReadFailed` when fstat fails.
     */
    static func fileByteCount(fileDescriptor: Int32) throws -> Int {
        var fileStatistics: stat = stat()
        let statisticsStatus: Int32 = fstat(fileDescriptor, &fileStatistics)
        guard statisticsStatus == 0 else {
            throw MlxRuntimeError.positionalReadFailed(
                description: "fstat failed: \(String(cString: strerror(errno)))")
        }
        return Int(fileStatistics.st_size)
    }

    /**
     * Assembles `headerBytes` followed by every payload range into one
     * contiguous, heap-allocated buffer returned as zero-copy `Data`. The
     * Data owns the buffer and deallocates it when the last reference dies,
     * which is what lets `SafetensorsFile.assembledBytes` keep eager arrays
     * addressable.
     *
     * - Parameters:
     *   - fileDescriptor: the open descriptor all ranges are read from.
     *   - headerBytes: bytes placed at the front of the assembled buffer.
     *   - payloadRequests: source ranges placed after the header bytes.
     *   - maximumConcurrentReadCount: the ceiling of overlapping reads; at
     *     least one is always honored.
     *   - metrics: optional instrumentation; each chunk read is measured.
     * - Throws: `MlxRuntimeError.positionalReadFailed` on the first failing
     *   chunk read; sibling reads still drain before the throw happens.
     */
    static func read(
        fileDescriptor: Int32,
        headerBytes: Data,
        payloadRequests: [ReadRequest],
        maximumConcurrentReadCount: Int,
        metrics: PositionalFileReadMetrics?
    ) throws -> Data {
        let totalByteCount: Int = headerBytes.count
            + payloadRequests.reduce(0, { (accumulatedByteCount: Int, readRequest: ReadRequest) -> Int in
                return accumulatedByteCount + readRequest.byteCount
            })
        guard totalByteCount > 0 else {
            return Data()
        }

        let assembledBuffer: UnsafeMutableRawBufferPointer =
            UnsafeMutableRawBufferPointer.allocate(byteCount: totalByteCount, alignment: ASSEMBLED_BUFFER_ALIGNMENT)
        headerBytes.withUnsafeBytes({ (rawHeaderBytes: UnsafeRawBufferPointer) -> Void in
            assembledBuffer.copyMemory(from: rawHeaderBytes)
        })

        let assembledBufferBox: AssembledBufferBox = AssembledBufferBox(buffer: assembledBuffer)
        let readChunks: [ReadChunk] = payloadRequests.flatMap({ (readRequest: ReadRequest) -> [ReadChunk] in
            return chunkRequest(readRequest)
        })

        let effectiveMaximumConcurrentReadCount: Int = max(1, maximumConcurrentReadCount)
        let readQueue: DispatchQueue = DispatchQueue(
            label: "astronomical.runtime-integration.positional-read",
            attributes: [.concurrent])
        let readPermitSemaphore: DispatchSemaphore = DispatchSemaphore(value: effectiveMaximumConcurrentReadCount)
        let readGroup: DispatchGroup = DispatchGroup()
        let firstFailureBox: FirstFailureBox = FirstFailureBox()

        for readChunk in readChunks {
            readGroup.enter()
            readQueue.async(execute: { () -> Void in
                let permitStatus: DispatchTimeoutResult = readPermitSemaphore.wait(timeout: .distantFuture)
                defer {
                    readPermitSemaphore.signal()
                    readGroup.leave()
                }
                guard permitStatus == .success else {
                    firstFailureBox.recordIfFirst("the positional read permit was not granted")
                    return
                }
                let failureDescription: String? = Self.runMeasuredRead(
                    metrics: metrics,
                    byteCount: readChunk.byteCount,
                    chunkReadOperation: { () -> String? in
                        return Self.performChunkRead(
                            fileDescriptor: fileDescriptor,
                            readChunk: readChunk,
                            destinationBuffer: assembledBufferBox.buffer)
                    })
                if let failureDescription: String = failureDescription {
                    firstFailureBox.recordIfFirst(failureDescription)
                }
            })
        }
        readGroup.wait()

        if let failureDescription: String = firstFailureBox.firstFailure() {
            assembledBuffer.deallocate()
            throw MlxRuntimeError.positionalReadFailed(description: failureDescription)
        }
        guard let assembledBaseAddress: UnsafeMutableRawPointer = assembledBuffer.baseAddress else {
            assembledBuffer.deallocate()
            throw MlxRuntimeError.positionalReadFailed(
                description: "the assembled buffer of \(totalByteCount) bytes has no base address")
        }
        return Data(
            bytesNoCopy: assembledBaseAddress,
            count: totalByteCount,
            deallocator: .custom({ (pointer: UnsafeMutableRawPointer, byteCount: Int) -> Void in
                UnsafeMutableRawBufferPointer(start: pointer, count: byteCount).deallocate()
            }))
    }

    /**
     * Splits one request into sequential pieces of at most
     * `CHUNK_BYTE_COUNT`, advancing both offsets per piece.
     */
    private static func chunkRequest(_ readRequest: ReadRequest) -> [ReadChunk] {
        var readChunks: [ReadChunk] = []
        var sourceFileOffset: UInt64 = readRequest.sourceFileOffset
        var destinationBufferOffset: Int = readRequest.destinationBufferOffset
        var remainingByteCount: Int = readRequest.byteCount
        while (remainingByteCount > 0) {
            let pieceByteCount: Int = min(CHUNK_BYTE_COUNT, remainingByteCount)
            readChunks.append(ReadChunk(
                sourceFileOffset: sourceFileOffset,
                destinationBufferOffset: destinationBufferOffset,
                byteCount: pieceByteCount))
            sourceFileOffset += UInt64(pieceByteCount)
            destinationBufferOffset += pieceByteCount
            remainingByteCount -= pieceByteCount
        }
        return readChunks
    }

    /**
     * Performs one chunk read as a pread loop that survives EINTR
     * (interrupted syscall) retries and short reads.
     *
     * - Returns: nil on success, or a human-readable failure description.
     */
    private static func performChunkRead(
        fileDescriptor: Int32,
        readChunk: ReadChunk,
        destinationBuffer: UnsafeMutableRawBufferPointer
    ) -> String? {
        guard let destinationBaseAddress: UnsafeMutableRawPointer = destinationBuffer.baseAddress else {
            return "the destination buffer has no base address"
        }
        var destinationCursor: UnsafeMutableRawPointer =
            destinationBaseAddress.advanced(by: readChunk.destinationBufferOffset)
        var sourceFileOffset: UInt64 = readChunk.sourceFileOffset
        var remainingByteCount: Int = readChunk.byteCount
        while (remainingByteCount > 0) {
            let readByteCount: Int = pread(
                fileDescriptor,
                destinationCursor,
                remainingByteCount,
                off_t(clamping: sourceFileOffset))
            if (readByteCount < 0) {
                if (errno == EINTR) {
                    continue
                }
                return "pread failed: \(String(cString: strerror(errno)))"
            }
            if (readByteCount == 0) {
                return "pread reached end of file with \(remainingByteCount) bytes still unread"
            }
            destinationCursor = destinationCursor.advanced(by: readByteCount)
            sourceFileOffset += UInt64(readByteCount)
            remainingByteCount -= readByteCount
        }
        return nil
    }

    /**
     * Runs the chunk read unwrapped when no metrics are attached, or
     * measured for overlap, latency, and volume when they are.
     */
    private static func runMeasuredRead(
        metrics: PositionalFileReadMetrics?,
        byteCount: Int,
        chunkReadOperation: () -> String?
    ) -> String? {
        guard let metrics: PositionalFileReadMetrics = metrics else {
            return chunkReadOperation()
        }
        var failureDescription: String? = nil
        _ = metrics.measureRead(byteCount: byteCount, operation: { () -> Bool in
            failureDescription = chunkReadOperation()
            return failureDescription == nil
        })
        return failureDescription
    }

    /**
     * Keeps the first recorded read failure; concurrent readers race to
     * record, and the first failure is the diagnostic that matters.
     *
     * Marked `@unchecked Sendable`: the single field is lock-guarded.
     */
    private final class FirstFailureBox: @unchecked Sendable {

        private let stateLock: NSLock = NSLock()
        private var failureDescription: String? = nil

        func recordIfFirst(_ description: String) {
            stateLock.lock()
            if (failureDescription == nil) {
                failureDescription = description
            }
            stateLock.unlock()
        }

        func firstFailure() -> String? {
            stateLock.lock()
            defer {
                stateLock.unlock()
            }
            return failureDescription
        }
    }

    /**
     * Carries the assembled buffer across concurrency domains; the raw
     * buffer type itself is not Sendable.
     */
    private final class AssembledBufferBox: @unchecked Sendable {

        let buffer: UnsafeMutableRawBufferPointer

        init(buffer: UnsafeMutableRawBufferPointer) {
            self.buffer = buffer
        }
    }
}
