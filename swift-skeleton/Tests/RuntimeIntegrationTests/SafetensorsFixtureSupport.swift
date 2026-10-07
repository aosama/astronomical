import Foundation

/**
 * Shared SafeTensors test fixtures: collision-proof temporary file URLs
 * and little-endian byte encodings for hand-built safetensors payloads.
 * One home for the byte-layout helpers every SafeTensors suite needs, so
 * each suite stays focused on its own journey.
 */
enum SafetensorsFixtureSupport {

    static func littleEndianBytes(of values: [Float]) -> [UInt8] {
        return values.flatMap({ (elementValue: Float) -> [UInt8] in
            let elementBitPattern: UInt32 = elementValue.bitPattern
            return [
                UInt8((elementBitPattern >> 0) & 0xFF),
                UInt8((elementBitPattern >> 8) & 0xFF),
                UInt8((elementBitPattern >> 16) & 0xFF),
                UInt8((elementBitPattern >> 24) & 0xFF),
            ]
        })
    }

    static func littleEndianLengthPrefix(of value: UInt64) -> [UInt8] {
        return stride(from: 0, to: 64, by: 8).map({ (bitShift: Int) -> UInt8 in
            return UInt8((value >> bitShift) & 0xFF)
        })
    }

    static func temporaryFileUrl(_ fileName: String) -> URL {
        return URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("safetensors-" + UUID().uuidString + "-" + fileName)
    }
}
