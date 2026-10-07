import Foundation

import CryptoKit

/// Content digests the Library publication flow verifies against: SHA-256
/// for LFS-backed files, Git blob SHA-1 for plain repository files.
public enum LibraryDownloadFileDigest {

    public static func sha256Hex(_ bytes: Data) -> String {
        let digest: SHA256.Digest = SHA256.hash(data: bytes)
        return digest.map({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte)
        }).joined()
    }

    public static func gitBlobSha1Hex(_ bytes: Data) -> String {
        var blobContent: Data = Data("blob \(bytes.count)\0".utf8)
        blobContent.append(bytes)
        let digest: Insecure.SHA1.Digest = Insecure.SHA1.hash(data: blobContent)
        return digest.map({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte)
        }).joined()
    }
}
