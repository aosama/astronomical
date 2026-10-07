import Foundation

import IpcProtocol

/**
 * Provider-supplied integrity evidence retained across manifest discovery
 * and durable jobs. The wire shape is the Hugging Face storage
 * representation: a tagged algorithm plus lowercase hex digest.
 */
public enum DownloadFileDigest: Equatable, Sendable {
    case sha256(String)
    case gitBlobSha1(String)

    private static let SHA256_HEX_CHARACTER_COUNT: Int = 64
    private static let GIT_BLOB_SHA1_HEX_CHARACTER_COUNT: Int = 40

    public var isValid: Bool {
        switch self {
        case let .sha256(hexDigest):
            return DownloadFileDigest.isLowercaseHex(
                hexDigest,
                expectedCharacterCount: DownloadFileDigest.SHA256_HEX_CHARACTER_COUNT)
        case let .gitBlobSha1(hexDigest):
            return DownloadFileDigest.isLowercaseHex(
                hexDigest,
                expectedCharacterCount: DownloadFileDigest.GIT_BLOB_SHA1_HEX_CHARACTER_COUNT)
        }
    }

    public var hex: String {
        switch self {
        case let .sha256(hexDigest):
            return hexDigest
        case let .gitBlobSha1(hexDigest):
            return hexDigest
        }
    }

    /// The tagged wire representation the durable job document carries.
    public var wireValue: JsonWireValue {
        var digestObject: JsonWireObject = JsonWireObject(entries: Array())
        switch self {
        case .sha256:
            digestObject.appendEntry(key: "algorithm", value: .string("sha256"))
        case .gitBlobSha1:
            digestObject.appendEntry(key: "algorithm", value: .string("git_blob_sha1"))
        }
        digestObject.appendEntry(key: "hex", value: .string(self.hex))
        return .object(digestObject)
    }

    public static func isLowercaseHex(_ digest: String, expectedCharacterCount: Int) -> Bool {
        guard digest.count == expectedCharacterCount else {
            return false
        }
        return digest.allSatisfy({ (digestCharacter: Character) -> Bool in
            return ("0"..."9").contains(digestCharacter) || ("a"..."f").contains(digestCharacter)
        })
    }
}
