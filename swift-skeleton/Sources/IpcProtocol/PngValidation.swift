import Foundation;

/// Structural facts extracted from a fully validated PNG completion payload.
internal struct ValidatedPng: Equatable {
    internal let widthPixels: UInt32;
    internal let heightPixels: UInt32;
    internal let decodedRgbByteCount: Int;

    internal init(widthPixels: UInt32, heightPixels: UInt32, decodedRgbByteCount: Int) {
        self.widthPixels = widthPixels;
        self.heightPixels = heightPixels;
        self.decodedRgbByteCount = decodedRgbByteCount;
    }
}

/// Performs allocation-free structural validation of untrusted PNG completion
/// payloads: every chunk is CRC-checked before it is interpreted, geometry is
/// read only from the first IHDR, and the decoded byte budget is computed from
/// header arithmetic before any pixel decoder allocates memory.
internal enum PngValidation {
    internal static let pngMimeType: String = "image/png";

    private static let pngSignatureBytes: Array<UInt8> = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    private static let chunkHeaderBytes: Int = 8;
    private static let chunkCrcBytes: Int = 4;
    private static let ihdrDataBytes: Int = 13;
    private static let requiredBitDepth: UInt8 = 8;
    private static let truecolorColorType: UInt8 = 2;
    private static let ihdrTypeBytes: Array<UInt8> = [0x49, 0x48, 0x44, 0x52];
    private static let idatTypeBytes: Array<UInt8> = [0x49, 0x44, 0x41, 0x54];
    private static let iendTypeBytes: Array<UInt8> = [0x49, 0x45, 0x4E, 0x44];

    internal static func validatePngStructure(pngBytes: Array<UInt8>, maximumDecodedRgbBytes: Int) throws -> ValidatedPng {
        if pngBytes.count < PngValidation.pngSignatureBytes.count {
            throw ImageGenerationCompletionValidationError.invalidPngEncoding;
        }
        for signatureIndex in 0..<PngValidation.pngSignatureBytes.count {
            if pngBytes[signatureIndex] != PngValidation.pngSignatureBytes[signatureIndex] {
                throw ImageGenerationCompletionValidationError.invalidPngEncoding;
            }
        }

        var chunkStart: Int = PngValidation.pngSignatureBytes.count;
        var widthPixels: UInt32? = nil;
        var heightPixels: UInt32? = nil;
        var sourceBitDepth: UInt8? = nil;
        var sourceColorType: UInt8? = nil;
        var idatDataByteCount: Int = 0;

        while chunkStart < pngBytes.count {
            let (chunkHeaderEnd, headerOverflowed) = chunkStart.addingReportingOverflow(PngValidation.chunkHeaderBytes);
            if headerOverflowed || chunkHeaderEnd > pngBytes.count {
                throw ImageGenerationCompletionValidationError.invalidPngEncoding;
            }
            let chunkDataByteCount: Int = Int(PngValidation.readBigEndianUInt32(pngBytes, at: chunkStart));
            let chunkDataStart: Int = chunkStart + PngValidation.chunkHeaderBytes;
            let (chunkDataEnd, dataEndOverflowed) = chunkDataStart.addingReportingOverflow(chunkDataByteCount);
            if dataEndOverflowed {
                throw ImageGenerationCompletionValidationError.invalidPngEncoding;
            }
            let (chunkEnd, chunkEndOverflowed) = chunkDataEnd.addingReportingOverflow(PngValidation.chunkCrcBytes);
            if chunkEndOverflowed || chunkEnd > pngBytes.count {
                throw ImageGenerationCompletionValidationError.invalidPngEncoding;
            }

            let chunkTypeStart: Int = chunkStart + 4;
            let expectedCrc: UInt32 = PngValidation.readBigEndianUInt32(pngBytes, at: chunkDataEnd);
            let crcMatches: Bool = PngValidation.crcOverBytes(
                pngBytes: pngBytes,
                ranges: [chunkTypeStart..<chunkDataStart, chunkDataStart..<chunkDataEnd],
                expectedCrc: expectedCrc);
            if crcMatches == false {
                throw ImageGenerationCompletionValidationError.invalidPngEncoding;
            }

            let chunkTypeBytes: Array<UInt8> = Array(pngBytes[chunkTypeStart..<chunkDataStart]);
            if chunkTypeBytes == PngValidation.ihdrTypeBytes {
                if chunkStart != PngValidation.pngSignatureBytes.count
                    || chunkDataByteCount != PngValidation.ihdrDataBytes
                    || widthPixels != nil {
                    throw ImageGenerationCompletionValidationError.invalidPngEncoding;
                }
                widthPixels = PngValidation.readBigEndianUInt32(pngBytes, at: chunkDataStart);
                heightPixels = PngValidation.readBigEndianUInt32(pngBytes, at: chunkDataStart + 4);
                sourceBitDepth = pngBytes[chunkDataStart + 8];
                sourceColorType = pngBytes[chunkDataStart + 9];
            } else if chunkTypeBytes == PngValidation.idatTypeBytes {
                let (accumulatedIdatBytes, idatOverflowed) = idatDataByteCount.addingReportingOverflow(chunkDataByteCount);
                if idatOverflowed {
                    throw ImageGenerationCompletionValidationError.invalidPngEncoding;
                }
                idatDataByteCount = accumulatedIdatBytes;
            } else if chunkTypeBytes == PngValidation.iendTypeBytes {
                if chunkDataByteCount != 0 || chunkEnd != pngBytes.count || idatDataByteCount == 0 {
                    throw ImageGenerationCompletionValidationError.invalidPngEncoding;
                }
                guard let encodedWidthPixels = widthPixels, let encodedHeightPixels = heightPixels else {
                    throw ImageGenerationCompletionValidationError.invalidPngEncoding;
                }
                if sourceBitDepth != PngValidation.requiredBitDepth || sourceColorType != PngValidation.truecolorColorType {
                    // Decoder output is insufficient evidence because palette PNGs expand to RGB8.
                    throw ImageGenerationCompletionValidationError.nonRgb8Png;
                }
                let decodedRgbByteCount: Int = try PngValidation.decodedRgbByteCount(
                    widthPixels: encodedWidthPixels, heightPixels: encodedHeightPixels);
                if decodedRgbByteCount > maximumDecodedRgbBytes {
                    throw ImageGenerationCompletionValidationError.pngDecodeResourceLimit;
                }
                return ValidatedPng(
                    widthPixels: encodedWidthPixels,
                    heightPixels: encodedHeightPixels,
                    decodedRgbByteCount: decodedRgbByteCount);
            }
            chunkStart = chunkEnd;
        }
        throw ImageGenerationCompletionValidationError.invalidPngEncoding;
    }

    private static func decodedRgbByteCount(widthPixels: UInt32, heightPixels: UInt32) throws -> Int {
        let (pixelCount, pixelCountOverflowed) = Int(widthPixels).multipliedReportingOverflow(by: Int(heightPixels));
        if pixelCountOverflowed {
            throw ImageGenerationCompletionValidationError.pngDecodeResourceLimit;
        }
        let (rgbByteCount, rgbByteCountOverflowed) = pixelCount.multipliedReportingOverflow(by: 3);
        if rgbByteCountOverflowed {
            throw ImageGenerationCompletionValidationError.pngDecodeResourceLimit;
        }
        return rgbByteCount;
    }

    private static func readBigEndianUInt32(_ pngBytes: Array<UInt8>, at byteOffset: Int) -> UInt32 {
        return (UInt32(pngBytes[byteOffset]) << 24)
            | (UInt32(pngBytes[byteOffset + 1]) << 16)
            | (UInt32(pngBytes[byteOffset + 2]) << 8)
            | UInt32(pngBytes[byteOffset + 3]);
    }

    private static func crcOverBytes(pngBytes: Array<UInt8>, ranges: Array<Range<Int>>, expectedCrc: UInt32) -> Bool {
        var crcState: UInt32 = 0xFFFF_FFFF;
        for byteRange in ranges {
            for byteIndex in byteRange {
                crcState = PngValidation.crcTable[Int((crcState ^ UInt32(pngBytes[byteIndex])) & 0xFF)] ^ (crcState >> 8);
            }
        }
        return (crcState ^ 0xFFFF_FFFF) == expectedCrc;
    }

    private static let crcTable: Array<UInt32> = PngValidation.buildCrcTable();

    private static func buildCrcTable() -> Array<UInt32> {
        var tableEntries: Array<UInt32> = Array<UInt32>();
        tableEntries.reserveCapacity(256);
        for tableIndex: UInt32 in 0..<256 {
            var tableEntry: UInt32 = tableIndex;
            for _ in 0..<8 {
                if tableEntry & 1 != 0 {
                    tableEntry = (tableEntry >> 1) ^ 0xEDB8_8320;
                } else {
                    tableEntry = tableEntry >> 1;
                }
            }
            tableEntries.append(tableEntry);
        }
        return tableEntries;
    }
}
