import Foundation

/**
 * Bounded arguments payload for one tool call, mirroring the Rust record from
 * apps/supervisor/src/completion_attribution_log.rs.
 *
 * `sizeBytes` and `sha256` are always present so identical payloads correlate
 * regardless of truncation. `json` is the full arguments string when at or
 * under the cap, or a truncated preview when over the cap; `truncated`
 * distinguishes the two.
 */
public struct CompletionArgumentsRecord: Equatable {

    /// Maximum number of bytes of arguments JSON recorded verbatim. Arguments
    /// at or under this cap are written in full so polluted keys (`content`,
    /// `hash`, duplicates) are visible. Larger arguments are truncated and
    /// hashed so the log never grows without bound while still correlating
    /// identical payloads.
    public static let ARGUMENTS_FULL_LIMIT_BYTES: Int = 8_192

    /// Original argument length in UTF-8 bytes, before any truncation.
    public let sizeBytes: Int

    /// SHA-256 of the full original arguments, never of the truncation.
    public let sha256: String

    /// The arguments JSON: full when `truncated` is false, a bounded preview
    /// when `truncated` is true.
    public let json: String

    /// Whether `json` is a truncation rather than the full arguments.
    public let truncated: Bool

    /**
     * Builds one bounded payload from the raw arguments JSON.
     *
     * Arguments at or under the cap are recorded verbatim. Larger arguments
     * are truncated to the cap at a UTF-8 scalar boundary and the full
     * original is hashed so identical payloads correlate regardless of
     * truncation.
     */
    public static func fromArgumentsJson(_ argumentsJson: String) -> CompletionArgumentsRecord {
        let argumentsBytes: Array<UInt8> = Array(argumentsJson.utf8)
        let sizeBytes: Int = argumentsBytes.count
        let sha256Hex: String = LibraryDownloadFileDigest.sha256Hex(Data(argumentsBytes))
        if sizeBytes <= CompletionArgumentsRecord.ARGUMENTS_FULL_LIMIT_BYTES {
            return CompletionArgumentsRecord(
                sizeBytes: sizeBytes,
                sha256: sha256Hex,
                json: argumentsJson,
                truncated: false)
        }
        let truncationBoundary: Int = CompletionArgumentsRecord.utf8ScalarBoundary(
            atMost: CompletionArgumentsRecord.ARGUMENTS_FULL_LIMIT_BYTES,
            in: argumentsBytes)
        let truncatedJson: String = String(
            decoding: argumentsBytes[0..<truncationBoundary],
            as: UTF8.self)
        return CompletionArgumentsRecord(
            sizeBytes: sizeBytes,
            sha256: sha256Hex,
            json: truncatedJson,
            truncated: true)
    }

    /// Backs off to the nearest UTF-8 scalar boundary at or below the byte
    /// limit so a truncation never splits one Unicode scalar.
    private static func utf8ScalarBoundary(atMost byteLimit: Int, in argumentsBytes: Array<UInt8>) -> Int {
        var boundary: Int = byteLimit
        while boundary > 0 {
            let candidateByte: UInt8 = argumentsBytes[boundary - 1]
            // Continuation bytes are 0b10xxxxxx; a scalar never ends on one.
            if candidateByte & 0xC0 != 0x80 {
                return boundary
            }
            boundary = boundary - 1
        }
        return 0
    }
}
